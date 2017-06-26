
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
#include <opencv2/calib3d.hpp>

#define DATASETS_VAP_USE_OLD_CALIB_DATA     0
#define DATASETS_VAP_FIX_GT_SCENE2_DISTORT  1
#define DATASETS_VAP_FIX_GT_SCENE3_OFFSET   1

namespace lv {

    /// parameter interface for VAP trimodal dataset loader impl
    struct IVAPtrimod2016Dataset {
        /// returns whether the depth stream should be loaded & provided as input or not (if not, the dataset is used as a bimodal one)
        virtual bool isLoadingDepth() const = 0;
        /// returns whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        virtual bool isUndistorting() const = 0;
        /// returns whether images should be horizontally rectified when loaded or not, using the calib files provided with the dataset
        virtual bool isHorizRectifying() const = 0;
        /// returns whether we should evaluate fg/bg segmentation or stereo disparities
        virtual bool isEvaluatingDisparities() const = 0;
        /// returns whether only a subset of the dataset's frames will be loaded or not
        virtual bool isLoadingFrameSubset() const = 0;
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        virtual int isLoadingInputMasks() const = 0;
    };

    /// dataset loader impl specialization for VAP trimodal dataset -- instantiated via lv::datasets::create(...)
    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_VAP_trimod2016,eEvalImpl> :
            public IVAPtrimod2016Dataset,
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAP_trimod2016,lv::getDatasetEval<eDatasetTask,Dataset_VAP_trimod2016>(),eEvalImpl> {
    protected: // should still be protected, as creation should always be done via lv::datasets::create
        /// specialization constructor, with all required extra parameters; these will be forwarded by lv::datasets::create(...)
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in VAP trimodal dataset results folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bLoadDepth=true, ///< defines whether the depth stream should be loaded or not (if not, the dataset is used as a bimodal one)
                bool bUndistort=true, ///< defines whether images should be undistorted when loaded or not, using the calib files provided with the dataset
                bool bHorizRectify=false, ///< defines whether images should be horizontally rectified when loaded or not, using the calib files provided with the dataset
                bool bEvalDisparities=false, ///< defines whether we should evaluate fg/bg segmentation or stereo disparities
                bool bLoadFrameSubset=false, ///< defines whether only a subset of the dataset's frames will be loaded or not
                int nLoadInputMasks=0, ///< defines whether the input stream should be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
                double dScaleFactor=1.0 ///< defines the scale factor to use to resize/rescale read packets
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAP_trimod2016,lv::getDatasetEval<eDatasetTask,Dataset_VAP_trimod2016>(),eEvalImpl>(
                        "VAP-trimodal2016",
                        lv::datasets::getRootPath()+"vap/rgbdt-stereo/",
                        DataHandler::createOutputDir(lv::datasets::getRootPath()+"vap/rgbdt-stereo/results/",sOutputDirName),
                        getWorkBatchDirNames(bUndistort||bHorizRectify),
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
                m_bLoadFrameSubset(bLoadFrameSubset),
                m_nLoadInputMasks(nLoadInputMasks) {
            lvAssert_(!m_bEvalDisparities,"missing impl (no stereo disparity gt in dataset)");
        }
        /// returns the names of all work batch directories available for this dataset specialization
        static const std::vector<std::string>& getWorkBatchDirNames(bool bUndistortOrRectify) {
            // dataset contains 3 scenes with GT for default use
            static const std::vector<std::string> s_vsWorkBatchDirs_real = {
                "Scene 1",
                "Scene 2",
                "Scene 3"
            };
            // 'Scene 2' does not have proper calibration data; must skip it if undistort/rectify required
            static const std::vector<std::string> s_vsWorkBatchDirs_calibonly = {
                "Scene 1",
                "Scene 3"
            };
            return bUndistortOrRectify?s_vsWorkBatchDirs_calibonly:s_vsWorkBatchDirs_real;
        }
        /// returns whether the depth stream should be loaded & provided as input or not (if not, the dataset is used as a bimodal one)
        virtual bool isLoadingDepth() const override {return m_bLoadDepth;}
        /// returns whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        virtual bool isUndistorting() const override {return m_bUndistort;}
        /// returns whether images should be horizontally rectified when loaded or not, using the calib files provided with the dataset
        virtual bool isHorizRectifying() const override {return m_bHorizRectify;}
        /// returns whether we should evaluate fg/bg segmentation or stereo disparities
        virtual bool isEvaluatingDisparities() const override {return m_bEvalDisparities;}
        /// returns whether only a subset of the dataset's frames will be loaded or not
        virtual bool isLoadingFrameSubset() const override {return m_bLoadFrameSubset;}
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        virtual int isLoadingInputMasks() const override {return m_nLoadInputMasks;}
    protected:
        const bool m_bLoadDepth;
        const bool m_bUndistort;
        const bool m_bHorizRectify;
        const bool m_bEvalDisparities;
        const bool m_bLoadFrameSubset;
        const int m_nLoadInputMasks;
    };

    /// data grouper handler impl specialization for vap trimodal dataset; will skip groups entirely & forward data to batches
    template<DatasetTaskList eDatasetTask>
    struct DataGroupHandler_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAP_trimod2016> :
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

    /// data producer impl specialization for vap trimodal dataset; provides required data i/o and undistort support
    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAP_trimod2016> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAP_trimod2016> {
        /// returns the number of parallel input streams (depends on whether loading depth or not)
        virtual size_t getInputStreamCount() const override final {
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
                const std::string sBaseName = (((nStreamIdx/2)==0)?"RGB":((nStreamIdx/2)==1)?"THERMAL":((nStreamIdx/2)==2)?"DEPTH":"UNKNOWN");
                return sBaseName+std::string((nStreamIdx%2)?"_MASK":"");
            }
            else
                return ((nStreamIdx==0)?"RGB":(nStreamIdx==1)?"THERMAL":(nStreamIdx==2)?"DEPTH":"UNKNOWN");
        }
        /// returns the (friendly) name of a gt stream specified by index
        virtual std::string getGTStreamName(size_t nStreamIdx) const override final {
            lvAssert(nStreamIdx<getGTStreamCount());
            const std::string sBaseName = ((nStreamIdx==0)?"RGB":(nStreamIdx==1)?"THERMAL":(nStreamIdx==2)?"DEPTH":"UNKNOWN");
            return sBaseName+"_GT";
        }
        /// returns whether the depth stream should be loaded & provided as input or not (if not, the dataset is used as a bimodal one)
        bool isLoadingDepth() const {
            return dynamic_cast<const IVAPtrimod2016Dataset&>(*this->getRoot()).isLoadingDepth();
        }
        /// returns whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        bool isUndistorting() const {
            return dynamic_cast<const IVAPtrimod2016Dataset&>(*this->getRoot()).isUndistorting();
        }
        /// returns whether images should be horizontally rectified when loaded or not, using the calib files provided with the dataset
        bool isHorizRectifying() const {
            return dynamic_cast<const IVAPtrimod2016Dataset&>(*this->getRoot()).isHorizRectifying();
        }
        /// returns whether we should evaluate fg/bg segmentation or stereo disparities
        bool isEvaluatingDisparities() const {
            return dynamic_cast<const IVAPtrimod2016Dataset&>(*this->getRoot()).isEvaluatingDisparities();
        }
        /// returns whether only a subset of the dataset's frames will be loaded or not
        bool isLoadingFrameSubset() const {
            return dynamic_cast<const IVAPtrimod2016Dataset&>(*this->getRoot()).isLoadingFrameSubset();
        }
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        int isLoadingInputMasks() const {
            return dynamic_cast<const IVAPtrimod2016Dataset&>(*this->getRoot()).isLoadingInputMasks();
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
            const IVAPtrimod2016Dataset& oDataset = dynamic_cast<const IVAPtrimod2016Dataset&>(*this->getRoot());
            const bool bLoadFrameSubset = this->isLoadingFrameSubset();
            this->m_bLoadDepth = oDataset.isLoadingDepth();
            this->m_nLoadInputMasks = oDataset.isLoadingInputMasks();
            this->m_bUndistort = oDataset.isUndistorting();
            this->m_bHorizRectify = oDataset.isHorizRectifying();
            lvAssert(!this->m_bHorizRectify || this->m_bUndistort);
            const bool bUseInterlacedMasks = this->m_nLoadInputMasks!=0;
            const bool bUseApproxRGBMask = (this->m_nLoadInputMasks&2)==0;
            const bool bUseApproxThermalMask = (this->m_nLoadInputMasks&4)==0;
            const bool bUseApproxDepthMask = (this->m_nLoadInputMasks&8)==0;
            const size_t nInputStreamCount = this->getInputStreamCount();
            const size_t nGTStreamCount = this->getGTStreamCount();
            constexpr size_t nInputRGBStreamIdx = 0;
            const size_t nInputThermalStreamIdx = bUseInterlacedMasks?2:1;
            const size_t nInputDepthStreamIdx = bUseInterlacedMasks?4:2;
            constexpr size_t nInputRGBMaskStreamIdx = 1;
            constexpr size_t nInputThermalMaskStreamIdx = 3;
            constexpr size_t nInputDepthMaskStreamIdx = 5;
            constexpr size_t nGTRGBMaskStreamIdx = 0;
            constexpr size_t nGTThermalMaskStreamIdx = 1;
            constexpr size_t nGTDepthMaskStreamIdx = 2;
            const std::vector<std::string> vsSubDirs = lv::getSubDirsFromDir(this->getDataPath());
            auto psRGBGTMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"rgbMasks");
            auto psRGBApproxMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"rgbApproxMasks");
            auto psRGBDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"SyncRGB");
            auto psThermalGTMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"thermalMasks");
            auto psThermalApproxMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"thermalApproxMasks");
            auto psThermalDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"SyncT");
            auto psDepthGTMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"depthMasks");
            auto psDepthApproxMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"depthApproxMasks");
            auto psDepthDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"SyncD");
            if((psDepthDir==vsSubDirs.end() || psDepthGTMasksDir==vsSubDirs.end()) || (psRGBDir==vsSubDirs.end() || psRGBGTMasksDir==vsSubDirs.end()) || (psThermalDir==vsSubDirs.end() || psThermalGTMasksDir==vsSubDirs.end()))
                lvError_("VAPtrimod2016 sequence '%s' did not possess the required groundtruth and input directories",this->getName().c_str());
            if(bUseInterlacedMasks) {
                if(bUseApproxRGBMask && psRGBApproxMasksDir==vsSubDirs.end())
                    lvError_("VAPtrimod2016 sequence '%s' did not possess the required approx input mask directory ('rgbApproxMasks')",this->getName().c_str());
                if(bUseApproxThermalMask && psThermalApproxMasksDir==vsSubDirs.end())
                    lvError_("VAPtrimod2016 sequence '%s' did not possess the required approx input mask directory ('thermalApproxMasks')",this->getName().c_str());
                if(this->m_bLoadDepth && bUseApproxDepthMask && psDepthApproxMasksDir==vsSubDirs.end())
                    lvError_("VAPtrimod2016 sequence '%s' did not possess the required approx input mask directory ('depthApproxMasks')",this->getName().c_str());
            }
            std::set<size_t> mSubset;
            if(bLoadFrameSubset) {
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
                    lvAssert(nCurrIdx>=0 && nCurrIdx<2500);
                    mSubset.insert(size_t(nCurrIdx));
                }
                lvAssert(mSubset.size()>0);
            }
            const auto lSubsetCleaner = [&](std::vector<std::string>& vsPaths) {
                lvAssert(!mSubset.empty() && vsPaths.size()>*mSubset.rbegin());
                std::vector<std::string> vsKeptPaths;
                for(const auto& nIdx : mSubset)
                    vsKeptPaths.push_back(vsPaths[nIdx]);
                vsPaths = vsKeptPaths;
            };
            this->m_vInputROIs.resize(nInputStreamCount);
            this->m_vGTROIs.resize(nGTStreamCount);
            this->m_vInputInfos.resize(nInputStreamCount);
            this->m_vGTInfos.resize(nGTStreamCount);
            const cv::Size oImageSize(640,480);
            cv::Mat oGlobalROI(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
            const double dScale = this->getScaleFactor();
            if(dScale!=1.0)
                cv::resize(oGlobalROI,oGlobalROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
            for(size_t nStreamIdx=0; nStreamIdx<nInputStreamCount; ++nStreamIdx)
                this->m_vInputInfos[nStreamIdx] = lv::MatInfo{oGlobalROI.size(),(nStreamIdx==nInputRGBStreamIdx?CV_8UC3:CV_8UC1)};
            for(size_t nStreamIdx=0; nStreamIdx<nGTStreamCount; ++nStreamIdx)
                this->m_vGTInfos[nStreamIdx] = lv::MatInfo{oGlobalROI.size(),CV_8UC1};
            if(this->m_bUndistort) {
            #if DATASETS_VAP_USE_OLD_CALIB_DATA
                lvAssert_(!this->m_bHorizRectify,"missing calib data for rectification");
                const double dUndistortMapCameraMatrixAlpha = -1.0;
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
            #else //!DATASETS_VAP_USE_OLD_CALIB_DATA
                const std::string sCalibDataFilePath = this->getDataPath()+"calib/calibdata.yml";
                cv::FileStorage oParamsFS(sCalibDataFilePath,cv::FileStorage::READ);
                lvAssert__(oParamsFS.isOpened(),"could not open calibration yml file at '%s'",sCalibDataFilePath.c_str());
                oParamsFS["aCamMats0"] >> this->m_oRGBCameraParams;
                oParamsFS["aDistCoeffs0"] >> this->m_oRGBDistortParams;
                lvAssert_(!this->m_oRGBCameraParams.empty() && !this->m_oRGBDistortParams.empty(),"failed to load RGB camera calibration parameters");
                oParamsFS["aCamMats1"] >> this->m_oThermalCameraParams;
                oParamsFS["aDistCoeffs1"] >> this->m_oThermalDistortParams;
                lvAssert_(!this->m_oThermalCameraParams.empty() && !this->m_oThermalDistortParams.empty(),"failed to load thermal camera calibration parameters");
                if(this->m_bHorizRectify) {
                    lvAssert_(!this->m_bLoadDepth,"missing depth image rectification impl");
                    std::array<cv::Mat,2> aRectifRotMats,aRectifProjMats;
                    cv::Mat oDispToDepthMap,oRotMat,oTranslMat;
                    oParamsFS["oRotMat"] >> oRotMat;
                    oParamsFS["oTranslMat"] >> oTranslMat;
                    lvAssert_(!this->m_oThermalCameraParams.empty() && !this->m_oThermalDistortParams.empty(),"failed to load thermal camera calibration parameters");
                    cv::stereoRectify(this->m_oRGBCameraParams,this->m_oRGBDistortParams,
                                      this->m_oThermalCameraParams,this->m_oThermalDistortParams,
                                      oImageSize,oRotMat,oTranslMat,
                                      aRectifRotMats[0],aRectifRotMats[1],
                                      aRectifProjMats[0],aRectifProjMats[1],
                                      oDispToDepthMap,
                                      0,//cv::CALIB_ZERO_DISPARITY,
                                      -1,cv::Size());
                    cv::initUndistortRectifyMap(this->m_oRGBCameraParams,this->m_oRGBDistortParams,
                                                aRectifRotMats[0],aRectifProjMats[0],oImageSize,
                                                CV_16SC2,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2);
                    cv::initUndistortRectifyMap(this->m_oThermalCameraParams,this->m_oThermalDistortParams,
                                                aRectifRotMats[1],aRectifProjMats[1],oImageSize,
                                                CV_16SC2,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2);
                }
                else {
                    const double dUndistortMapCameraMatrixAlpha = -1.0;
                    cv::initUndistortRectifyMap(this->m_oRGBCameraParams,this->m_oRGBDistortParams,cv::Mat(),
                                                (dUndistortMapCameraMatrixAlpha<0)?this->m_oRGBCameraParams:cv::getOptimalNewCameraMatrix(this->m_oRGBCameraParams,this->m_oRGBDistortParams,oImageSize,dUndistortMapCameraMatrixAlpha,oImageSize,0),
                                                oImageSize,CV_16SC2,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2);
                    cv::initUndistortRectifyMap(this->m_oThermalCameraParams,this->m_oThermalDistortParams,cv::Mat(),
                                                (dUndistortMapCameraMatrixAlpha<0)?this->m_oThermalCameraParams:cv::getOptimalNewCameraMatrix(this->m_oThermalCameraParams,this->m_oThermalDistortParams,oImageSize,dUndistortMapCameraMatrixAlpha,oImageSize,0),
                                                oImageSize,CV_16SC2,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2);
                }
            #endif //!DATASETS_VAP_USE_OLD_CALIB_DATA
            }
            this->m_nMinDisp = size_t(0);
            this->m_nMaxDisp = size_t(100);
            this->m_nThermalDispOffset = 0;
            std::ifstream oDispRangeFile(this->getDataPath()+"drange.txt");
            if(oDispRangeFile.is_open() && !oDispRangeFile.eof()) {
                oDispRangeFile >> this->m_nMinDisp;
                if(!oDispRangeFile.eof())
                    oDispRangeFile >> this->m_nMaxDisp;
            }
            if(this->m_nMinDisp>size_t(0)) {
                this->m_nThermalDispOffset = int(this->m_nMinDisp);
                this->m_nMaxDisp -= this->m_nMinDisp;
                this->m_nMinDisp = size_t(0);
            }
            this->m_nThermalDispOffset *= dScale;
            this->m_nMinDisp *= dScale;
            this->m_nMaxDisp *= dScale;
            lvAssert(this->m_nMaxDisp>this->m_nMinDisp);
            //////////////////////////////////////////////////////////////////////////////////////////////////
            std::vector<std::string> vsRGBPaths = lv::getFilesFromDir(*psRGBDir);
            if(vsRGBPaths.empty() || cv::imread(vsRGBPaths[0]).size()!=oImageSize)
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected RGB data",this->getName().c_str());
            if(bLoadFrameSubset)
                lSubsetCleaner(vsRGBPaths);
            const size_t nInputPackets = vsRGBPaths.size();
            this->m_vvsInputPaths.resize(nInputPackets);
            std::vector<std::string> vsTempInputFileNames(nInputPackets);
            for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx) {
                this->m_vvsInputPaths[nInputPacketIdx].resize(nInputStreamCount);
                this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx] = vsRGBPaths[nInputPacketIdx];
                const size_t nLastInputSlashPos = vsRGBPaths[nInputPacketIdx].find_last_of("/\\");
                const std::string sInputFileNameExt = nLastInputSlashPos==std::string::npos?vsRGBPaths[nInputPacketIdx]:vsRGBPaths[nInputPacketIdx].substr(nLastInputSlashPos+1);
                const size_t nLastInputDotPos = sInputFileNameExt.find_last_of('.');
                vsTempInputFileNames[nInputPacketIdx] = nLastInputDotPos==std::string::npos?sInputFileNameExt:sInputFileNameExt.substr(0,nLastInputDotPos);
            }
            if(bUseInterlacedMasks && bUseApproxRGBMask) {
                std::vector<std::string> vsRGBApproxMasksPaths = lv::getFilesFromDir(*psRGBApproxMasksDir);
                if(vsRGBApproxMasksPaths.empty() || cv::imread(vsRGBApproxMasksPaths[0]).size()!=oImageSize)
                    lvError_("VAPtrimod2016 sequence '%s' did not possess expected RGB approx mask data",this->getName().c_str());
                if(bLoadFrameSubset)
                    lSubsetCleaner(vsRGBApproxMasksPaths);
                lvAssert_(vsRGBApproxMasksPaths.size()==nInputPackets,"rgb approx mask count did not match input image count");
                for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx)
                    this->m_vvsInputPaths[nInputPacketIdx][nInputRGBMaskStreamIdx] = vsRGBApproxMasksPaths[nInputPacketIdx];
            }
            cv::Mat oRGBROI = cv::imread(this->getDataPath()+"rgb_roi.png",cv::IMREAD_GRAYSCALE);
            if(!oRGBROI.empty()) {
                lvAssert(oRGBROI.type()==CV_8UC1 && oRGBROI.size()==oImageSize);
                oRGBROI = oRGBROI>128;
            }
            else
                oRGBROI = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
            if(oRGBROI.size()!=this->m_vInputInfos[nInputRGBStreamIdx].size()) {
                cv::resize(oRGBROI,oRGBROI,this->m_vInputInfos[nInputRGBStreamIdx].size(),0,0,cv::INTER_LINEAR);
                oRGBROI = oRGBROI>128;
            }
            if(this->m_bUndistort || this->m_bHorizRectify) {
                cv::remap(oRGBROI.clone(),oRGBROI,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_LINEAR);
                oRGBROI = oRGBROI>128;
                cv::erode(oRGBROI,oRGBROI,cv::Mat(),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar_<uchar>(0));
                cv::Mat oRGBROI_undist = cv::imread(this->getDataPath()+"rgb_undist_roi.png",cv::IMREAD_GRAYSCALE);
                if(!oRGBROI_undist.empty()) {
                    lvAssert(oRGBROI_undist.type()==CV_8UC1 && oRGBROI_undist.size()==oImageSize);
                    oRGBROI_undist = oRGBROI_undist>128;
                }
                else
                    oRGBROI_undist = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
                if(oRGBROI_undist.size()!=this->m_vInputInfos[nInputRGBStreamIdx].size()) {
                    cv::resize(oRGBROI_undist,oRGBROI_undist,this->m_vInputInfos[nInputRGBStreamIdx].size(),0,0,cv::INTER_LINEAR);
                    oRGBROI_undist = oRGBROI_undist>128;
                }
                oRGBROI &= oRGBROI_undist;
            }
            this->m_vInputROIs[nInputRGBStreamIdx] = oRGBROI.clone();
            if(bUseInterlacedMasks)
                this->m_vInputROIs[nInputRGBMaskStreamIdx] = oRGBROI.clone();
            this->m_vGTROIs[nGTRGBMaskStreamIdx] = oRGBROI.clone();
            std::vector<std::string> vsRGBGTMasksPaths = lv::getFilesFromDir(*psRGBGTMasksDir);
            if(vsRGBGTMasksPaths.empty() || cv::imread(vsRGBGTMasksPaths[0]).size()!=oImageSize)
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected RGB gt data",this->getName().c_str());
            if(bLoadFrameSubset)
                lSubsetCleaner(vsRGBGTMasksPaths);
            const size_t nGTPackets = vsRGBGTMasksPaths.size();
            this->m_vvsGTPaths.resize(nGTPackets);
            this->m_mGTIndexLUT.clear();
            for(size_t nGTPacketIdx=0; nGTPacketIdx<nGTPackets; ++nGTPacketIdx) {
                this->m_vvsGTPaths[nGTPacketIdx].resize(nGTStreamCount);
                this->m_vvsGTPaths[nGTPacketIdx][nGTRGBMaskStreamIdx] = vsRGBGTMasksPaths[nGTPacketIdx];
                const size_t nLastGTSlashPos = vsRGBGTMasksPaths[nGTPacketIdx].find_last_of("/\\");
                const std::string sGTFileNameExt = nLastGTSlashPos==std::string::npos?vsRGBGTMasksPaths[nGTPacketIdx]:vsRGBGTMasksPaths[nGTPacketIdx].substr(nLastGTSlashPos+1);
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
            if(bUseInterlacedMasks && !bUseApproxRGBMask) {
                lvAssert_(vsRGBGTMasksPaths.size()==nInputPackets,"rgb gt mask count did not match input image count");
                for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx)
                    this->m_vvsInputPaths[nInputPacketIdx][nInputRGBMaskStreamIdx] = vsRGBGTMasksPaths[nInputPacketIdx];
            }
            //////////////////////////////////////////////////////////////////////////////////////////
            std::vector<std::string> vsThermalPaths = lv::getFilesFromDir(*psThermalDir);
            if(vsThermalPaths.empty() || cv::imread(vsThermalPaths[0]).size()!=oImageSize)
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected thermal data",this->getName().c_str());
            if(bLoadFrameSubset)
                lSubsetCleaner(vsThermalPaths);
            if(vsThermalPaths.size()!=nInputPackets)
                lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/thermal frames",this->getName().c_str());
            for(size_t nInputPacketIdx=0; nInputPacketIdx<vsThermalPaths.size(); ++nInputPacketIdx)
                this->m_vvsInputPaths[nInputPacketIdx][nInputThermalStreamIdx] = vsThermalPaths[nInputPacketIdx];
            if(bUseInterlacedMasks && bUseApproxThermalMask) {
                std::vector<std::string> vsThermalApproxMasksPaths = lv::getFilesFromDir(*psThermalApproxMasksDir);
                if(vsThermalApproxMasksPaths.empty() || cv::imread(vsThermalApproxMasksPaths[0]).size()!=oImageSize)
                    lvError_("VAPtrimod2016 sequence '%s' did not possess expected thermal approx mask data",this->getName().c_str());
                if(bLoadFrameSubset)
                    lSubsetCleaner(vsThermalApproxMasksPaths);
                lvAssert_(vsThermalApproxMasksPaths.size()==nInputPackets,"thermal approx mask count did not match input image count");
                for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx)
                    this->m_vvsInputPaths[nInputPacketIdx][nInputThermalMaskStreamIdx] = vsThermalApproxMasksPaths[nInputPacketIdx];
            }
            cv::Mat oThermalROI = cv::imread(this->getDataPath()+"thermal_roi.png",cv::IMREAD_GRAYSCALE);
            if(!oThermalROI.empty()) {
                lvAssert(oThermalROI.type()==CV_8UC1 && oThermalROI.size()==oImageSize);
                oThermalROI = oThermalROI>128;
            }
            else
                oThermalROI = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
            if(oThermalROI.size()!=this->m_vInputInfos[nInputThermalStreamIdx].size()) {
                cv::resize(oThermalROI,oThermalROI,this->m_vInputInfos[nInputThermalStreamIdx].size(),0,0,cv::INTER_LINEAR);
                oThermalROI = oThermalROI>128;
            }
            if(this->m_bUndistort || this->m_bHorizRectify) {
                cv::remap(oThermalROI.clone(),oThermalROI,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2,cv::INTER_LINEAR);
                if(this->m_nThermalDispOffset!=0)
                    lv::shift(oThermalROI.clone(),oThermalROI,cv::Point2f(0.0f,-float(this->m_nThermalDispOffset)));
                oThermalROI = oThermalROI>128;
                cv::erode(oThermalROI,oThermalROI,cv::Mat(),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar_<uchar>(0));
                cv::Mat oThermalROI_undist = cv::imread(this->getDataPath()+"thermal_undist_roi.png",cv::IMREAD_GRAYSCALE);
                if(!oThermalROI_undist.empty()) {
                    lvAssert(oThermalROI_undist.type()==CV_8UC1 && oThermalROI_undist.size()==oImageSize);
                    oThermalROI_undist = oThermalROI_undist>128;
                }
                else
                    oThermalROI_undist = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
                if(oThermalROI_undist.size()!=this->m_vInputInfos[nInputRGBStreamIdx].size()) {
                    cv::resize(oThermalROI_undist,oThermalROI_undist,this->m_vInputInfos[nInputRGBStreamIdx].size(),0,0,cv::INTER_LINEAR);
                    oThermalROI_undist = oThermalROI_undist>128;
                }
                oThermalROI &= oThermalROI_undist;
            }
            this->m_vInputROIs[nInputThermalStreamIdx] = oThermalROI.clone();
            if(bUseInterlacedMasks)
                this->m_vInputROIs[nInputThermalMaskStreamIdx] = oThermalROI.clone();
            this->m_vGTROIs[nGTThermalMaskStreamIdx] = oThermalROI.clone();
            std::vector<std::string> vsThermalGTMasksPaths = lv::getFilesFromDir(*psThermalGTMasksDir);
            if(vsThermalGTMasksPaths.empty() || cv::imread(vsThermalGTMasksPaths[0]).size()!=oImageSize)
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected thermal gt data",this->getName().c_str());
            if(bLoadFrameSubset)
                lSubsetCleaner(vsThermalGTMasksPaths);
            if(vsThermalGTMasksPaths.size()!=nGTPackets)
                lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/thermal gt frames",this->getName().c_str());
            for(size_t nGTPacketIdx=0; nGTPacketIdx<vsThermalGTMasksPaths.size(); ++nGTPacketIdx)
                this->m_vvsGTPaths[nGTPacketIdx][nGTThermalMaskStreamIdx] = vsThermalGTMasksPaths[nGTPacketIdx];
            if(bUseInterlacedMasks && !bUseApproxThermalMask) {
                lvAssert_(vsThermalGTMasksPaths.size()==nInputPackets,"thermal gt mask count did not match input image count");
                for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx)
                    this->m_vvsInputPaths[nInputPacketIdx][nInputThermalMaskStreamIdx] = vsThermalGTMasksPaths[nInputPacketIdx];
            }
            //////////////////////////////////////////////////////////////////////////////////////////
            if(this->m_bLoadDepth) {
                std::vector<std::string> vsDepthPaths = lv::getFilesFromDir(*psDepthDir);
                if(vsDepthPaths.empty() || cv::imread(vsDepthPaths[0]).size()!=oImageSize)
                    lvError_("VAPtrimod2016 sequence '%s' did not possess expected depth data",this->getName().c_str());
                if(bLoadFrameSubset)
                    lSubsetCleaner(vsDepthPaths);
                if(vsDepthPaths.size()!=nInputPackets)
                    lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/depth frames",this->getName().c_str());
                for(size_t nInputPacketIdx=0; nInputPacketIdx<vsDepthPaths.size(); ++nInputPacketIdx)
                    this->m_vvsInputPaths[nInputPacketIdx][nInputDepthStreamIdx] = vsDepthPaths[nInputPacketIdx];
                if(bUseInterlacedMasks && bUseApproxDepthMask) {
                    std::vector<std::string> vsDepthApproxMasksPaths = lv::getFilesFromDir(*psDepthApproxMasksDir);
                    if(vsDepthApproxMasksPaths.empty() || cv::imread(vsDepthApproxMasksPaths[0]).size()!=oImageSize)
                        lvError_("VAPtrimod2016 sequence '%s' did not possess expected depth approx mask data",this->getName().c_str());
                    if(bLoadFrameSubset)
                        lSubsetCleaner(vsDepthApproxMasksPaths);
                    lvAssert_(vsDepthApproxMasksPaths.size()==nInputPackets,"depth approx mask count did not match input image count");
                    for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx)
                        this->m_vvsInputPaths[nInputPacketIdx][nInputDepthMaskStreamIdx] = vsDepthApproxMasksPaths[nInputPacketIdx];
                }
                cv::Mat oDepthROI = cv::imread(this->getDataPath()+"depth_roi.png",cv::IMREAD_GRAYSCALE);
                if(!oDepthROI.empty()) {
                    lvAssert(oDepthROI.type()==CV_8UC1 && oDepthROI.size()==oImageSize);
                    oDepthROI = oDepthROI>128;
                }
                else
                    oDepthROI = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
                if(oDepthROI.size()!=this->m_vInputInfos[nInputDepthStreamIdx].size()) {
                    cv::resize(oDepthROI,oDepthROI,this->m_vInputInfos[nInputDepthStreamIdx].size(),0,0,cv::INTER_LINEAR);
                    oDepthROI = oDepthROI>128;
                }
                this->m_vInputROIs[nInputDepthStreamIdx] = oDepthROI.clone();
                if(bUseInterlacedMasks)
                    this->m_vInputROIs[nInputDepthMaskStreamIdx] = oDepthROI.clone();
                this->m_vGTROIs[nGTDepthMaskStreamIdx] = oDepthROI.clone();
                std::vector<std::string> vsDepthGTMasksPaths = lv::getFilesFromDir(*psDepthGTMasksDir);
                if(vsDepthGTMasksPaths.empty() || cv::imread(vsDepthGTMasksPaths[0]).size()!=oImageSize)
                    lvError_("VAPtrimod2016 sequence '%s' did not possess expected depth gt data",this->getName().c_str());
                if(bLoadFrameSubset)
                    lSubsetCleaner(vsDepthGTMasksPaths);
                if(vsDepthGTMasksPaths.size()!=nGTPackets)
                    lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/depth gt frames",this->getName().c_str());
                for(size_t nGTPacketIdx=0; nGTPacketIdx<vsDepthGTMasksPaths.size(); ++nGTPacketIdx)
                    this->m_vvsGTPaths[nGTPacketIdx][nGTDepthMaskStreamIdx] = vsDepthGTMasksPaths[nGTPacketIdx];
                if(bUseInterlacedMasks && !bUseApproxDepthMask) {
                    lvAssert_(vsDepthGTMasksPaths.size()==nInputPackets,"depth gt mask count did not match input image count");
                    for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx)
                        this->m_vvsInputPaths[nInputPacketIdx][nInputDepthMaskStreamIdx] = vsDepthGTMasksPaths[nInputPacketIdx];
                }
            }
            //////////////////////////////////////////////////////////////////////////////////////////////////
            this->m_vOrigInputInfos = this->m_vInputInfos;
            this->m_vOrigGTInfos = this->m_vGTInfos;
            if(this->m_bHorizRectify) {
                if(this->m_bLoadDepth)
                    lvAssert_(false,"missing impl");
                for(size_t nIdx=0; nIdx<this->m_vInputROIs.size(); ++nIdx) {
                    cv::transpose(this->m_vInputROIs[nIdx],this->m_vInputROIs[nIdx]);
                    cv::flip(this->m_vInputROIs[nIdx],this->m_vInputROIs[nIdx],1);
                    this->m_vInputInfos[nIdx].size = this->m_vInputInfos[nIdx].size.transpose();
                }
                for(size_t nIdx=0; nIdx<this->m_vGTROIs.size(); ++nIdx) {
                    cv::transpose(this->m_vGTROIs[nIdx],this->m_vGTROIs[nIdx]);
                    cv::flip(this->m_vGTROIs[nIdx],this->m_vGTROIs[nIdx],1);
                    this->m_vGTInfos[nIdx].size = this->m_vGTInfos[nIdx].size.transpose();
                }
            }
        }
        virtual std::vector<cv::Mat> getRawInputArray(size_t nPacketIdx) override final {
            lvDbgExceptionWatch;
            if(nPacketIdx>=this->m_vvsInputPaths.size())
                return std::vector<cv::Mat>(this->getInputStreamCount());
            const cv::Size oImageSize(640,480);
            const bool bUseInterlacedMasks = this->m_nLoadInputMasks!=0;
            constexpr size_t nInputRGBStreamIdx = 0;
            const size_t nInputThermalStreamIdx = bUseInterlacedMasks?2:1;
            const size_t nInputDepthStreamIdx = bUseInterlacedMasks?4:2;
            constexpr size_t nInputRGBMaskStreamIdx = 1;
            constexpr size_t nInputThermalMaskStreamIdx = 3;
            constexpr size_t nInputDepthMaskStreamIdx = 5;
            const std::vector<std::string>& vsInputPaths = this->m_vvsInputPaths[nPacketIdx];
            lvDbgAssert(!vsInputPaths.empty() && vsInputPaths.size()==this->getInputStreamCount());
            const std::vector<lv::MatInfo>& vInputInfos = this->m_vOrigInputInfos;
            lvDbgAssert(!vInputInfos.empty() && vInputInfos.size()==getInputStreamCount());
            std::vector<cv::Mat> vInputs(vsInputPaths.size());
            ///////////////////////////////////////////////////////////////////////////////////
            cv::Mat oRGBPacket = cv::imread(vsInputPaths[nInputRGBStreamIdx],cv::IMREAD_COLOR);
            lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC3 && oRGBPacket.size()==oImageSize);
            if(oRGBPacket.size()!=vInputInfos[nInputRGBStreamIdx].size())
                cv::resize(oRGBPacket,oRGBPacket,vInputInfos[nInputRGBStreamIdx].size(),0,0,cv::INTER_CUBIC);
            if(this->m_bUndistort || this->m_bHorizRectify)
                cv::remap(oRGBPacket.clone(),oRGBPacket,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_CUBIC);
            vInputs[nInputRGBStreamIdx] = oRGBPacket;
            if(bUseInterlacedMasks) {
                cv::Mat oRGBMaskPacket = cv::imread(vsInputPaths[nInputRGBMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oRGBMaskPacket.empty() && oRGBMaskPacket.type()==CV_8UC1 && oRGBMaskPacket.size()==oImageSize);
                if(oRGBMaskPacket.size()!=vInputInfos[nInputRGBStreamIdx].size())
                    cv::resize(oRGBMaskPacket,oRGBMaskPacket,vInputInfos[nInputRGBStreamIdx].size(),cv::INTER_LINEAR);
                if(this->m_bUndistort || this->m_bHorizRectify)
                    cv::remap(oRGBMaskPacket.clone(),oRGBMaskPacket,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_LINEAR);
                oRGBMaskPacket = oRGBMaskPacket>128;
                vInputs[nInputRGBMaskStreamIdx] = oRGBMaskPacket;
            }
            ///////////////////////////////////////////////////////////////////////////////////
            cv::Mat oThermalPacket = cv::imread(vsInputPaths[nInputThermalStreamIdx],cv::IMREAD_GRAYSCALE);
            lvAssert(!oThermalPacket.empty() && oThermalPacket.type()==CV_8UC1 && oThermalPacket.size()==oImageSize);
            if(oThermalPacket.size()!=vInputInfos[nInputThermalStreamIdx].size())
                cv::resize(oThermalPacket,oThermalPacket,vInputInfos[nInputThermalStreamIdx].size(),0,0,cv::INTER_CUBIC);
            if(this->m_bUndistort || this->m_bHorizRectify) {
                cv::remap(oThermalPacket.clone(),oThermalPacket,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2,cv::INTER_CUBIC);
                if(this->m_nThermalDispOffset!=0)
                    lv::shift(oThermalPacket.clone(),oThermalPacket,cv::Point2f(0.0f,-float(this->m_nThermalDispOffset)));
            }
            vInputs[nInputThermalStreamIdx] = oThermalPacket;
            if(bUseInterlacedMasks) {
                cv::Mat oThermalMaskPacket = cv::imread(vsInputPaths[nInputThermalMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oThermalMaskPacket.empty() && oThermalMaskPacket.type()==CV_8UC1 && oThermalMaskPacket.size()==oImageSize);
                if(oThermalMaskPacket.size()!=vInputInfos[nInputThermalStreamIdx].size())
                    cv::resize(oThermalMaskPacket,oThermalMaskPacket,vInputInfos[nInputThermalStreamIdx].size(),cv::INTER_LINEAR);
                if(this->m_bUndistort || this->m_bHorizRectify) {
                    cv::remap(oThermalMaskPacket.clone(),oThermalMaskPacket,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2,cv::INTER_LINEAR);
                    if(this->m_nThermalDispOffset!=0)
                        lv::shift(oThermalMaskPacket.clone(),oThermalMaskPacket,cv::Point2f(0.0f,-float(this->m_nThermalDispOffset)));
                }
                oThermalMaskPacket = oThermalMaskPacket>128;
                vInputs[nInputThermalMaskStreamIdx] = oThermalMaskPacket;
            }
            ///////////////////////////////////////////////////////////////////////////////////
            if(this->m_bLoadDepth) {
                cv::Mat oDepthPacket = cv::imread(vsInputPaths[nInputDepthStreamIdx],cv::IMREAD_ANYDEPTH);
                lvAssert(!oDepthPacket.empty() && oDepthPacket.type()==CV_16UC1 && oDepthPacket.size()==oImageSize);
                if(oDepthPacket.size()!=vInputInfos[nInputDepthStreamIdx].size())
                    cv::resize(oDepthPacket,oDepthPacket,vInputInfos[nInputDepthStreamIdx].size(),0,0,cv::INTER_CUBIC);
                // depth should be already undistorted
                lvAssert_(!this->m_bHorizRectify,"missing depth image rectification impl");
                vInputs[nInputDepthStreamIdx] = oDepthPacket;
                if(bUseInterlacedMasks) {
                    cv::Mat oDepthMaskPacket = cv::imread(vsInputPaths[nInputDepthMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oDepthMaskPacket.empty() && oDepthMaskPacket.type()==CV_8UC1 && oDepthMaskPacket.size()==oImageSize);
                    if(oDepthMaskPacket.size()!=vInputInfos[nInputDepthStreamIdx].size())
                        cv::resize(oDepthMaskPacket,oDepthMaskPacket,vInputInfos[nInputDepthStreamIdx].size(),cv::INTER_LINEAR);
                    lvAssert_(!this->m_bHorizRectify,"missing depth image rectification impl");
                    oDepthMaskPacket = oDepthMaskPacket>128;
                    vInputs[nInputDepthMaskStreamIdx] = oDepthMaskPacket;
                }
            }
            if(this->m_bHorizRectify) {
                if(this->m_bLoadDepth)
                    lvAssert_(false,"missing impl");
                for(size_t nIdx=0; nIdx<vInputs.size(); ++nIdx) {
                    cv::transpose(vInputs[nIdx],vInputs[nIdx]);
                    cv::flip(vInputs[nIdx],vInputs[nIdx],1);
                    lvDbgAssert(lv::MatInfo(vInputs[nIdx])==this->m_vInputInfos[nIdx]);
                }
            }
            return vInputs;
        }
        virtual std::vector<cv::Mat> getRawGTArray(size_t nPacketIdx) override final {
            lvDbgExceptionWatch;
            const cv::Size oImageSize(640,480);
            constexpr size_t nGTRGBMaskStreamIdx = 0;
            constexpr size_t nGTThermalMaskStreamIdx = 1;
            constexpr size_t nGTDepthMaskStreamIdx = 2;
            if(this->m_mGTIndexLUT.count(nPacketIdx)) {
                const size_t nGTIdx = this->m_mGTIndexLUT[nPacketIdx];
                lvDbgAssert(nGTIdx<this->m_vvsGTPaths.size());
                const std::vector<std::string>& vsGTMasksPaths = this->m_vvsGTPaths[nGTIdx];
                lvDbgAssert(!vsGTMasksPaths.empty() && vsGTMasksPaths.size()==getGTStreamCount());
                const std::vector<lv::MatInfo>& vGTInfos = this->m_vOrigGTInfos;
                lvDbgAssert(!vGTInfos.empty() && vGTInfos.size()==getGTStreamCount());
                std::vector<cv::Mat> vGTs(vsGTMasksPaths.size());
                cv::Mat oRGBPacket = cv::imread(vsGTMasksPaths[nGTRGBMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC1 && oRGBPacket.size()==oImageSize);
                if(oRGBPacket.size()!=vGTInfos[nGTRGBMaskStreamIdx].size())
                    cv::resize(oRGBPacket,oRGBPacket,vGTInfos[nGTRGBMaskStreamIdx].size(),0,0,cv::INTER_LINEAR);
                if(this->m_bUndistort || this->m_bHorizRectify)
                    cv::remap(oRGBPacket.clone(),oRGBPacket,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_LINEAR);
                oRGBPacket = oRGBPacket>128;
                vGTs[nGTRGBMaskStreamIdx] = oRGBPacket;
                cv::Mat oThermalPacket = cv::imread(vsGTMasksPaths[nGTThermalMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oThermalPacket.empty() && oThermalPacket.type()==CV_8UC1 && oThermalPacket.size()==oImageSize);
#if DATASETS_VAP_FIX_GT_SCENE3_OFFSET
                // fail: calibration really breaks up for scene 3 (need to translate [x,y]=[13,4], and it's still not great)
                if(this->getName()=="Scene 3") {
                    static const cv::Mat_<double> oAffineTransf = (cv::Mat_<double>(2,3) << 1.0,0.0,13.0,0.0,1.0,4.0);
                    cv::warpAffine(oThermalPacket.clone(),oThermalPacket,oAffineTransf,oThermalPacket.size());
                }
#endif //DATASETS_VAP_FIX_GT_SCENE3_OFFSET
                if(oThermalPacket.size()!=vGTInfos[nGTThermalMaskStreamIdx].size())
                    cv::resize(oThermalPacket,oThermalPacket,vGTInfos[nGTThermalMaskStreamIdx].size(),0,0,cv::INTER_LINEAR);
#if DATASETS_VAP_FIX_GT_SCENE2_DISTORT
                // fail: 'distorted' thermal gt images are actually already undistorted in 'Scene 2' (so no need to remap again)
                if(this->getName()!="Scene 2")
#endif //DATASETS_VAP_FIX_GT_SCENE2_DISTORT
                {
                    if(this->m_bUndistort || this->m_bHorizRectify)
                        cv::remap(oThermalPacket.clone(),oThermalPacket,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2,cv::INTER_LINEAR);
                }
                if((this->m_bUndistort || this->m_bHorizRectify) && this->m_nThermalDispOffset!=0)
                    lv::shift(oThermalPacket.clone(),oThermalPacket,cv::Point2f(0.0f,-float(this->m_nThermalDispOffset)));
                oThermalPacket = oThermalPacket>128;
                vGTs[nGTThermalMaskStreamIdx] = oThermalPacket;
                if(this->m_bLoadDepth) {
                    cv::Mat oDepthPacket = cv::imread(vsGTMasksPaths[nGTDepthMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oDepthPacket.empty() && oDepthPacket.type()==CV_8UC1 && oDepthPacket.size()==oImageSize);
                    if(oDepthPacket.size()!=vGTInfos[nGTDepthMaskStreamIdx].size())
                        cv::resize(oDepthPacket,oDepthPacket,vGTInfos[nGTDepthMaskStreamIdx].size(),0,0,cv::INTER_LINEAR);
                    // depth should be already undistorted
                    lvAssert_(!this->m_bHorizRectify,"missing depth image rectification impl");
                    oDepthPacket = oDepthPacket>128;
                    vGTs[nGTDepthMaskStreamIdx] = oDepthPacket;
                }
                if(this->m_bHorizRectify) {
                    if(this->m_bLoadDepth)
                        lvAssert_(false,"missing impl");
                    for(size_t nIdx=0; nIdx<vGTs.size(); ++nIdx) {
                        cv::transpose(vGTs[nIdx],vGTs[nIdx]);
                        cv::flip(vGTs[nIdx],vGTs[nIdx],1);
                        lvDbgAssert(lv::MatInfo(vGTs[nIdx])==this->m_vGTInfos[nIdx]);
                    }
                }
                return vGTs;
            }
            return cv::Mat();
        }
        bool m_bLoadDepth,m_bUndistort,m_bHorizRectify;
        int m_nLoadInputMasks;
        int m_nThermalDispOffset;
        size_t m_nMinDisp,m_nMaxDisp;
        std::string m_sFeaturesDirName;
        cv::Mat m_oRGBCameraParams;
        cv::Mat m_oThermalCameraParams;
        cv::Mat m_oRGBDistortParams;
        cv::Mat m_oThermalDistortParams;
        cv::Mat m_oRGBCalibMap1,m_oRGBCalibMap2;
        cv::Mat m_oThermalCalibMap1,m_oThermalCalibMap2;
        std::vector<lv::MatInfo> m_vOrigInputInfos,m_vOrigGTInfos;
    };

} // namespace lv
