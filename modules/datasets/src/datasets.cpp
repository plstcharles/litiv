
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

#include "litiv/datasets.hpp"

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
        m_bForcingGrayscale(PlatformUtils::string_contains_token(sRelativePath,pDataset->getGrayscaleDirTokens())),
        m_pDataset(pDataset) {
    PlatformUtils::CreateDirIfNotExist(m_sOutputPath);
}

litiv::IDataHandlerConstPtr litiv::DataHandler::getBatch(size_t& nPacketIdx) const {
    if(isGroup()) {
        size_t nCurrPacketCount = 0;
        auto vpBatches = getBatches(true);
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
        return (*ppBatchIter)->shared_from_this_cast<DataHandler>(true)->getBatch(nPacketIdx);
    }
    else {
        CV_Assert(nPacketIdx<getTotPackets());
        return shared_from_this();
    }
}

litiv::IDataHandlerPtr litiv::DataHandler::getBatch(size_t& nPacketIdx) {
    return std::const_pointer_cast<IDataHandler>(static_cast<const DataHandler*>(this)->getBatch(nPacketIdx));
}
