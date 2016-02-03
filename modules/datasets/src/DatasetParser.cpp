
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

#include "litiv/datasets/DatasetParser.hpp"

const std::string& litiv::DataHandler::getName() const {return m_sBatchName;}
const std::string& litiv::DataHandler::getDataPath() const {return m_sDataPath;}
const std::string& litiv::DataHandler::getOutputPath() const {return m_sOutputPath;}
const std::string& litiv::DataHandler::getRelativePath() const {return m_sRelativePath;}
bool litiv::DataHandler::isGrayscale() const {return m_bForcingGrayscale;}
litiv::IDatasetPtr litiv::DataHandler::getDatasetInfo() const {return m_pDataset;}

litiv::DataHandler::DataHandler(const std::string& sBatchName, IDatasetPtr pDataset, const std::string& sRelativePath) :
        m_sBatchName(sBatchName),
        m_sRelativePath(sRelativePath),
        m_sDataPath(pDataset->getDatasetPath()+sRelativePath),
        m_sOutputPath(pDataset->getOutputPath()+sRelativePath),
        m_bForcingGrayscale(PlatformUtils::string_contains_token(sBatchName,pDataset->getGrayscaleDirTokens())),
        m_pDataset(pDataset) {
    PlatformUtils::CreateDirIfNotExist(m_sOutputPath);
}

litiv::IDataHandlerConstPtr litiv::DataHandler::getBatch(size_t& nPacketIdx) const {
        if(isGroup()) {
            size_t nCurrPacketCount = 0;
            auto vpBatches = getBatches();
            auto ppBatchIter = vpBatches.begin();
            while(ppBatchIter!=vpBatches.end()) {
                const size_t nNextPacketIncr = (*ppBatchIter)->getTotPackets();
                if(nPacketIdx<nCurrPacketCount+nNextPacketIncr)
                    break;
                nCurrPacketCount += nNextPacketIncr;
                ++ppBatchIter;
            }
            CV_Assert(ppBatchIter!=vpBatches.end());
            nPacketIdx -= nCurrPacketCount;
            return *ppBatchIter;
        }
        else {
            CV_Assert(nPacketIdx<getTotPackets());
            return shared_from_this();
        }
}

litiv::IDataHandlerPtr litiv::DataHandler::getBatch(size_t& nPacketIdx) {
    return std::const_pointer_cast<IDataHandler>(static_cast<const DataHandler*>(this)->getBatch(nPacketIdx));
}

void litiv::DataProducer_<litiv::eDatasetType_VideoSegm,litiv::eDataset_VideoSegm_CDnet,litiv::eNotGroup>::parseData() {
    std::vector<std::string> vsSubDirs;
    PlatformUtils::GetSubDirsFromDir(getDataPath(),vsSubDirs);
    auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),getDataPath()+"/groundtruth");
    auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),getDataPath()+"/input");
    if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
        lvErrorExt("CDnet Sequence '%s' did not possess the required groundtruth and input directories",getName().c_str());
    PlatformUtils::GetFilesFromDir(*inputDir,m_vsInputFramePaths);
    PlatformUtils::GetFilesFromDir(*gtDir,m_vsGTFramePaths);
    if(m_vsGTFramePaths.size()!=m_vsInputFramePaths.size())
        lvErrorExt("CDnet Sequence '%s' did not possess same amount of GT & input frames",getName().c_str());
    m_oROI = cv::imread(getDataPath()+"/ROI.bmp",cv::IMREAD_GRAYSCALE);
    if(m_oROI.empty())
        lvErrorExt("CDnet Sequence '%s' did not possess a ROI.bmp file",getName().c_str());
    m_oROI = m_oROI>0; // @@@@@ check throw here???
    m_oSize = m_oROI.size();
    m_nFrameCount = m_vsInputFramePaths.size();
    CV_Assert(m_nFrameCount>0);
    // note: in this case, no need to use m_vnTestGTIndexes since all # of gt frames == # of test frames (but we assume the frames returned by 'GetFilesFromDir' are ordered correctly...)
}

cv::Mat litiv::DataProducer_<litiv::eDatasetType_VideoSegm,litiv::eDataset_VideoSegm_CDnet,litiv::eNotGroup>::_getGTPacket_impl(size_t nIdx) {
    cv::Mat oFrame = cv::imread(m_vsGTFramePaths[nIdx],cv::IMREAD_GRAYSCALE);
    if(oFrame.empty())
        oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_VIDEOSEGM_OUTOFSCOPE_VAL));
    else if(oFrame.size()!=m_oSize)
        cv::resize(oFrame,oFrame,m_oSize,0,0,cv::INTER_NEAREST);
    return oFrame;
}

litiv::Dataset_<litiv::eDatasetType_VideoSegm,litiv::eDataset_VideoSegm_CDnet>::Dataset_(const std::string& sOutputDirName, bool bSaveOutput, bool bUseEvaluator, bool bForce4ByteDataAlign, double dScaleFactor, bool b2014) :
            IDataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet>(
                    b2014?"CDnet 2014":"CDnet 2012",
                    b2014?"CDNet2014/dataset":"CDNet/dataset",
                    std::string(DATASET_ROOT)+"/"+std::string(b2014?"CDNet2014/":"CDNet/")+sOutputDirName+"/",
                    "bin",
                    ".png",
                    std::vector<std::string>{"baseline_highway_cut2"},//b2014?std::vector<std::string>{"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"}:std::vector<std::string>{"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"},
                    std::vector<std::string>{},
                    b2014?std::vector<std::string>{"thermal","turbulence"}:std::vector<std::string>{"thermal"},
                    1,
                    bSaveOutput,
                    bUseEvaluator,
                    bForce4ByteDataAlign,
                    dScaleFactor
            ) {}
