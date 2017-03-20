
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

#include <stdlib.h>

#ifndef __LITIV_FGSTEREOM_HPP__
#error "Cannot include .inl.hpp headers directly!"
#endif //ndef(__LITIV_FGSTEREOM_HPP__)
#pragma once
#if (FGSTEREOMATCH_CONFIG_USE_DASC_FEATS+FGSTEREOMATCH_CONFIG_USE_LSS_FEATS/*+...*/)!=1
#error "Must specify only one feature type to use."
#endif //(features config ...)!=1

inline FGStereoMatcher::FGStereoMatcher(const cv::Size& oImageSize, int32_t nMinDispOffset, int32_t nMaxDispOffset, int32_t nDispStep) :
        m_oImageSize(oImageSize) {
    static_assert(getInputStreamCount()==4 && getOutputStreamCount()==4 && getCameraCount()==2,"i/o stream must be two image-mask pairs");
    lvAssert_(m_oImageSize.area()>1,"graph grid must be 2D and have at least two nodes");
    lvAssert_(nDispStep>0,"specified disparity offset step size must be strictly positive");
    if(nMaxDispOffset<nMinDispOffset)
        std::swap(nMaxDispOffset,nMinDispOffset);
    lvAssert_(nMinDispOffset>s_nStereoDontCareLabel,"using reserved disparity integer label value");
    lvAssert_(nMaxDispOffset<s_nStereoOccludedLabel,"using reserved disparity integer label value");
    const size_t nMaxAllowedDispLabelCount = size_t(std::numeric_limits<StereoLabelType>::max()-2);
    const size_t nExpectedDispLabelCount = size_t((nMaxDispOffset-nMinDispOffset)/nDispStep)+1;
    lvAssert__(nMaxAllowedDispLabelCount>=nExpectedDispLabelCount,"internal stereo label type too small for given disparity range (max = %d)",(int)nMaxAllowedDispLabelCount);
    const std::vector<OutputLabelType> vStereoLabels = lv::make_range(nMinDispOffset,nMaxDispOffset,nDispStep);
    lvDbgAssert(nExpectedDispLabelCount==vStereoLabels.size());
    lvAssert_(vStereoLabels.size()>1,"graph must have at least two possible output labels, beyond reserved ones");
    const std::vector<OutputLabelType> vReservedStereoLabels = {s_nStereoDontCareLabel,s_nStereoOccludedLabel};
#if FGSTEREOMATCH_CONFIG_USE_DASC_FEATS
    //std::unique_ptr<DASC> pFeatsExtractor = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR,DASC_DEFAULT_RF_ITERS,DASC_DEFAULT_PREPROCESS);
    std::unique_ptr<DASC> pFeatsExtractor = std::make_unique<DASC>(DASC_DEFAULT_GF_RADIUS,DASC_DEFAULT_GF_EPS,DASC_DEFAULT_GF_SUBSPL,DASC_DEFAULT_PREPROCESS);
#elif FGSTEREOMATCH_CONFIG_USE_LSS_FEATS
    @@@
#endif //FGSTEREOMATCH_CONFIG_USE_..._FEATS
    const cv::Size oMinWindowSize = pFeatsExtractor->windowSize();
    lvAssert__(oMinWindowSize.width<=oImageSize.width && oMinWindowSize.height<=oImageSize.height,"image is too small to compute descriptors with current pattern size -- need at least (%d,%d) and got (%d,%d)",oMinWindowSize.width,oMinWindowSize.height,oImageSize.width,oImageSize.height);
    m_pModelData = std::make_unique<GraphModelData>(m_oImageSize,std::move(pFeatsExtractor),lv::concat<OutputLabelType>(vStereoLabels,vReservedStereoLabels),size_t(nDispStep));
    m_pStereoInf = std::make_unique<StereoGraphInference>(*m_pModelData);
    //@@@@@ m_pResegmInf = std::make_unique<ResegmGraphInference>(*m_pModelData,m_pModelData->m_oResegmLabeling);
}

inline void FGStereoMatcher::apply(const MatArrayIn& aImages, MatArrayOut& oMasks) {
    for(size_t nImgIdx=0; nImgIdx<aImages.size(); ++nImgIdx) {
        lvAssert__(aImages[nImgIdx].dims==2 && m_oImageSize==aImages[nImgIdx].size(),"input image in array at index=%d had the wrong size",(int)nImgIdx);
        lvAssert_((nImgIdx%2)==0 || aImages[nImgIdx].type()==CV_8UC1,"unexpected input mask type");
        if((nImgIdx%2)==0 && aImages[nImgIdx].total()<100)
            std::cout << "aImages[" << nImgIdx << "] = " << lv::to_string(lv::getBasicMat<uchar>(aImages[nImgIdx])) << std::endl;
    }
    lvDbgAssert(m_pModelData && m_pStereoInf /*&& m_pResegmInf@@@@@*/);
    m_pModelData->updateModels(aImages);
    if(lv::getVerbosity()>=2) {
        StereoGraphInference::VerboseVisitorType oVisitor;
        m_pStereoInf->infer(oVisitor);
    }
    else {
        StereoGraphInference::EmptyVisitorType oVisitor;
        m_pStereoInf->infer(oVisitor);
    }
    m_pStereoInf->getOutput(oMasks[0]);
    // @@@@@ fill missing outputs
    for(size_t nImgIdx=1; nImgIdx<aImages.size(); ++nImgIdx)
        cv::Mat_<OutputLabelType>(aImages[nImgIdx].size(),OutputLabelType(0)).copyTo(oMasks[nImgIdx]);
}

inline void FGStereoMatcher::calcFeatures(const MatArrayIn& aImages, cv::Mat* pFeatsPacket) {
    m_pModelData->calcFeatures(aImages,pFeatsPacket);
}

inline void FGStereoMatcher::setNextFeatures(const cv::Mat& oPackedFeats) {
    m_pModelData->setNextFeatures(oPackedFeats);
}

inline std::string FGStereoMatcher::getFeatureExtractorName() const {
#if FGSTEREOMATCH_CONFIG_USE_DASC_FEATS
    return "dasc";
#elif FGSTEREOMATCH_CONFIG_USE_LSS_FEATS
    return "lss";
#endif //FGSTEREOMATCH_CONFIG_USE_..._FEATS
}

inline size_t FGStereoMatcher::getMaxLabelCount() const {
    lvDbgAssert(m_pModelData);
    return m_pModelData->m_vStereoLabels.size();
}

inline const std::vector<FGStereoMatcher::OutputLabelType>& FGStereoMatcher::getLabels() const {
    lvDbgAssert(m_pModelData);
    return m_pModelData->m_vStereoLabels;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename TFeatExtr>
inline FGStereoMatcher::GraphModelData::GraphModelData(const cv::Size& oImageSize, std::unique_ptr<TFeatExtr> pFeatExtr, std::vector<OutputLabelType>&& vStereoLabels, size_t nStereoLabelStep) :
        m_nMaxIterCount(FGSTEREOMATCH_DEFAULT_MAXITERCOUNT),
        m_eStereoLabelInitType(LabelInit_LocalOptim),
        m_eResegmLabelInitType(LabelInit_LocalOptim),
        m_eStereoLabelOrderType(LabelOrder_Default),
        m_eResegmLabelOrderType(LabelOrder_Default),
        m_nStereoLabelingRandomSeed(size_t(0)),
        m_nResegmLabelingRandomSeed(size_t(0)),
        m_nStereoLabelOrderRandomSeed(size_t(0)),
        m_nResegmLabelOrderRandomSeed(size_t(0)),
        m_oGridSize(oImageSize),
        m_nGridBorderSize((size_t)std::max(pFeatExtr->borderSize(0),pFeatExtr->borderSize(1))),
        m_nFeatMapsPerCam(3),
        m_vStereoLabels(vStereoLabels),
        m_nStereoLabelStep(nStereoLabelStep),
        m_nStereoDontCareLabelIdx(StereoLabelType(m_vStereoLabels.size()-2)),
        m_nStereoOccludedLabelIdx(StereoLabelType(m_vStereoLabels.size()-1)),
        m_pFeatExtractor(std::move(pFeatExtr)),
        m_bUsePrecalcFeatsNext(false),m_bModelUpToDate(false) {
    lvAssert_(m_nMaxIterCount>0,"max iter count must be positive");
    lvDbgAssert_(m_oGridSize.dims()==2 && m_oGridSize.total()>size_t(1),"graph grid must be 2D and have at least two nodes");
    lvAssert_(m_vStereoLabels.size()>3,"graph must have at least two possible output stereo labels, beyond reserved ones");
    lvAssert_(m_vStereoLabels.size()<=size_t(std::numeric_limits<StereoLabelType>::max()),"too many labels for internal type");
    lvDbgAssert(m_vStereoLabels[m_nStereoDontCareLabelIdx]==s_nStereoDontCareLabel && m_vStereoLabels[m_nStereoOccludedLabelIdx]==s_nStereoOccludedLabel);
    lvDbgAssert(std::min_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==m_vStereoLabels.begin());
    lvDbgAssert(std::max_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==(m_vStereoLabels.begin()+m_vStereoLabels.size()-3));
    lvDbgAssert(std::equal(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2,lv::unique(m_vStereoLabels.begin(),m_vStereoLabels.end()).begin()+1));
    lvAssert_(m_nStereoLabelStep>0,"label step size must be positive");
    lvDbgAssert_(std::numeric_limits<AssocCountType>::max()>m_oGridSize[1],"grid width is too large for association counter type");
    // @@@@@@ assert to make sure label count small enough to not make explicit func allocs go boom
    ////////////////////////////////////////////// put in new func, dupe for resegm model
    const size_t nLabels = vStereoLabels.size();
    const size_t nRows = m_oGridSize(0);
    const size_t nCols = m_oGridSize(1);
    const size_t nNodes = m_oGridSize.total();
    const size_t nStereoVisSimUnaryFuncDataSize = nNodes*nLabels;
    const size_t nStereoSmoothPairwFuncDataSize = nNodes*2*(nLabels*nLabels);
    const size_t nStereoFuncDataSize = nStereoVisSimUnaryFuncDataSize+nStereoSmoothPairwFuncDataSize/*+...@@@@*/;
    const size_t nModelSize = (nStereoFuncDataSize/*+...@@@@*/)*sizeof(ValueType)/*+...@@@@*/;
    lvLog_(1,"Expecting model size = %zu mb",nModelSize/1024/1024);
    lvAssert_(nModelSize<(CACHE_MAX_SIZE_GB*1024*1024*1024),"too many nodes/labels; model is unlikely to fit in memory");
    lvLog(1,"Constructing graphical model for stereo matching...");
    lv::StopWatch oLocalTimer;
    const size_t nFactorsPerNode = 3;
    const size_t nFunctions = nNodes*nFactorsPerNode;
    m_pStereoModel = std::make_unique<StereoModelType>(StereoSpaceType(nNodes,(StereoLabelType)vStereoLabels.size()),nFactorsPerNode);
    m_pStereoModel->reserveFunctions<StereoExplicitFunction>(nFunctions);
    const std::array<int,2> anAssocCountsDims{int(m_oGridSize[0]/m_nStereoLabelStep),int(m_oGridSize[1]/m_nStereoLabelStep)};
    m_oAssocCounts.create(2,anAssocCountsDims.data());
    const std::array<int,3> anAssocMapDims{int(m_oGridSize[0]/m_nStereoLabelStep),int(m_oGridSize[1]/m_nStereoLabelStep),int(m_oGridSize[1]/m_nStereoLabelStep)};
    m_oAssocMap.create(3,anAssocMapDims.data());
    m_vNodeInfos.resize(nNodes);
    for(int nRowIdx = 0; nRowIdx<(int)nRows; ++nRowIdx) {
        for(int nColIdx = 0; nColIdx<(int)nCols; ++nColIdx) {
            const size_t nNodeIdx = nRowIdx*nCols+nColIdx;
            m_vNodeInfos[nNodeIdx].nRowIdx = nRowIdx;
            m_vNodeInfos[nNodeIdx].nColIdx = nColIdx;
            // the LUT members below will be properly initialized in the following sections
            m_vNodeInfos[nNodeIdx].nStereoVisSimUnaryFactID = SIZE_MAX;
            m_vNodeInfos[nNodeIdx].anStereoSmoothPairwFactIDs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nNodeIdx].pStereoVisSimUnaryFunc = nullptr;
            m_vNodeInfos[nNodeIdx].apStereoSmoothPairwFuncs = std::array<StereoFunc*,2>{nullptr,nullptr};
        }
    }
    m_vStereoVisSimUnaryFuncs.reserve(nNodes);
    m_avStereoSmoothPairwFuncs[0].reserve(nNodes);
    m_avStereoSmoothPairwFuncs[1].reserve(nNodes);
#if FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
    m_pStereoVisSimUnaryFuncsDataBase = nullptr;
    m_pStereoSmoothPairwFuncsDataBase = nullptr;
#else //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
    m_aStereoFuncsData = std::make_unique<ValueType[]>(nStereoFuncDataSize);
    m_pStereoVisSimUnaryFuncsDataBase = m_aStereoFuncsData.get();
    m_pStereoSmoothPairwFuncsDataBase = m_pStereoVisSimUnaryFuncsDataBase+nStereoVisSimUnaryFuncDataSize;
#endif //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
    {
        lvLog(1,"\tadding visual similarity factor for each grid node...");
        // (unary costs will depend on input data of both images, so each node function is likely unique)
        const std::array<size_t,1> aUnaryFuncDims = {nLabels};
        for(size_t nNodeIdx = 0; nNodeIdx<nNodes; ++nNodeIdx) {
            if(m_vNodeInfos[nNodeIdx].nRowIdx<(int)m_nGridBorderSize || m_vNodeInfos[nNodeIdx].nRowIdx>=int(nRows-m_nGridBorderSize) ||
               m_vNodeInfos[nNodeIdx].nColIdx<(int)m_nGridBorderSize || m_vNodeInfos[nNodeIdx].nColIdx>=int(nCols-m_nGridBorderSize))
                continue; // @@@@@@ remove later, use border-friendly vissim updates instead
            m_vStereoVisSimUnaryFuncs.push_back(m_pStereoModel->addFunctionWithRefReturn(StereoExplicitFunction()));
            StereoFunc& oFunc = m_vStereoVisSimUnaryFuncs.back();
#if FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
            oFunc.second.resize(aUnaryFuncDims.begin(),aUnaryFuncDims.end());
#else //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
            oFunc.second.assign(aUnaryFuncDims.begin(),aUnaryFuncDims.end(),m_pStereoVisSimUnaryFuncsDataBase+(nNodeIdx*nLabels));
#endif //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
            lvDbgAssert(oFunc.second.strides(0)==1); // expect no padding
            const std::array<size_t,1> aNodeIndices = {nNodeIdx};
            m_vNodeInfos[nNodeIdx].nStereoVisSimUnaryFactID = m_pStereoModel->addFactorNonFinalized(oFunc.first,aNodeIndices.begin(),aNodeIndices.end());
            m_vNodeInfos[nNodeIdx].pStereoVisSimUnaryFunc = &oFunc;
        }
    }
    {
        lvLog(1,"\tadding label similarity factor for each grid node pair...");
        // note: current def w/ explicit function will require too much memory if using >>50 labels
        const std::array<size_t,2> aPairwiseFuncDims = {nLabels,nLabels};
        std::array<size_t,2> aNodeIndices;
        for(size_t nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(size_t nColIdx=0; nColIdx<nCols; ++nColIdx) {
                aNodeIndices[0] = nRowIdx*nCols+nColIdx;
                if(nRowIdx+1<nRows) { // vertical pair
                    aNodeIndices[1] = (nRowIdx+1)*nCols+nColIdx;
                    m_avStereoSmoothPairwFuncs[0].push_back(m_pStereoModel->addFunctionWithRefReturn(StereoExplicitFunction()));
                    StereoFunc& oFunc = m_avStereoSmoothPairwFuncs[0].back();
#if FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    oFunc.second.resize(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end());
#else //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    oFunc.second.assign(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end(),m_pStereoSmoothPairwFuncsDataBase+((aNodeIndices[0]*2)*nLabels*nLabels));
#endif //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    lvDbgAssert(oFunc.second.strides(0)==1 && oFunc.second.strides(1)==nLabels); // expect last-idx-major
                    m_vNodeInfos[aNodeIndices[0]].anStereoSmoothPairwFactIDs[0] = m_pStereoModel->addFactorNonFinalized(oFunc.first,aNodeIndices.begin(),aNodeIndices.end());
                    m_vNodeInfos[aNodeIndices[0]].apStereoSmoothPairwFuncs[0] = &oFunc;
                }
                if(nColIdx+1<nCols) { // horizontal pair
                    aNodeIndices[1] = nRowIdx*nCols+nColIdx+1;
                    m_avStereoSmoothPairwFuncs[1].push_back(m_pStereoModel->addFunctionWithRefReturn(StereoExplicitFunction()));
                    StereoFunc& oFunc = m_avStereoSmoothPairwFuncs[1].back();
#if FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    oFunc.second.resize(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end());
#else //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    oFunc.second.assign(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end(),m_pStereoSmoothPairwFuncsDataBase+((aNodeIndices[0]*2+1)*nLabels*nLabels));
#endif //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    lvDbgAssert(oFunc.second.strides(0)==1 && oFunc.second.strides(1)==nLabels); // expect last-idx-major
                    m_vNodeInfos[aNodeIndices[0]].anStereoSmoothPairwFactIDs[1] = m_pStereoModel->addFactorNonFinalized(oFunc.first,aNodeIndices.begin(),aNodeIndices.end());
                    m_vNodeInfos[aNodeIndices[0]].apStereoSmoothPairwFuncs[1] = &oFunc;
                }
            }
        }
    }/*{
     // add 3rd order function and factors to the model (test)
        const std::array<LabelType@,3> aHOEFuncDims = {nLabels,nLabels,nLabels};
        ExplicitFunction vHOEFunc(aHOEFuncDims.begin(),aHOEFuncDims.end(),0.5f);
        FunctionID nFID = m_pGM->addFunction(vHOEFunc);
        for(size_t nLabelIdx1=0; nLabelIdx1<nRealLabels; ++nLabelIdx1) {
            for(size_t nLabelIdx2 = 0; nLabelIdx2<nRealLabels; ++nLabelIdx2) {
                for(size_t nLabelIdx3 = 0; nLabelIdx3<nRealLabels; ++nLabelIdx3) {
                    ...
                }
            }
        }
        for(size_t nNodeIdx=nPxGridNodes; nNodeIdx<vNodeLabelCounts.size(); ++nNodeIdx) {
            const size_t nRandNodeIdx1 = ((rand()%nRows)*nCols+rand()%nCols);
            const size_t nRandNodeIdx2 = ((rand()%nRows)*nCols+rand()%nCols);
            const std::array<size_t,3> aNodeIndices = {nRandNodeIdx1<nRandNodeIdx2?nRandNodeIdx1:nRandNodeIdx2,nRandNodeIdx1<nRandNodeIdx2?nRandNodeIdx2:nRandNodeIdx1,nNodeIdx};
            m_pGM->addFactorNonFinalized(nFID,aNodeIndices.begin(),aNodeIndices.end());
        }
    }*/
    m_pStereoModel->finalize();
    lvLog_(1,"Stereo model constructed in %f second(s).",oLocalTimer.tock());
    lv::gm::printModelInfo(*m_pStereoModel);

    // @@@@ add resegm struct init here
    // @@@@@@@ RETEST RESEGM GRAPH STABILITY FROM SANDBOX W/ NEW STRUCTURE

}

inline void FGStereoMatcher::GraphModelData::resetLabelings() {
    lvDbgAssert_(m_oAssocMap.isContinuous() && m_oAssocCounts.isContinuous(),"stereo assoc maps must be continuous blocks");
    lvDbgAssert_(m_oAssocMap.step.p[0]==m_oGridSize[1]*m_oGridSize[1]*sizeof(AssocCheckType)/m_nStereoLabelStep,"unexpected assoc map row size");
    lvDbgAssert_(m_oAssocMap.step.p[1]==m_oGridSize[1]*sizeof(AssocCheckType)/m_nStereoLabelStep && m_oAssocMap.step.p[2]==sizeof(AssocCheckType),"unexpected assoc map col size");
    lvDbgAssert_(m_oAssocCounts.step.p[0]==m_oGridSize[1]*sizeof(AssocCountType)/m_nStereoLabelStep && m_oAssocCounts.step.p[1]==sizeof(AssocCountType),"unexpected assoc count map size");
    const StereoModelType& oGM = *m_pStereoModel;
    const size_t nLabels = m_vStereoLabels.size();
    const size_t nRealLabels = m_vStereoLabels.size()-2;
    const size_t nNodes = m_oGridSize.total();
    lvDbgAssert(nLabels>3 && nNodes==m_vNodeInfos.size());
    lvIgnore(oGM);
    if(m_eStereoLabelInitType==LabelInit_Default) {
        m_oStereoLabeling.create(m_oGridSize);
        std::fill(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),(StereoLabelType)0);
    }
    else if(m_eStereoLabelInitType==LabelInit_Random) {
        m_oStereoLabeling.create(m_oGridSize);
        std::mt19937 oGen(m_nStereoLabelingRandomSeed);
        std::generate(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),[&](){return StereoLabelType(oGen()%nRealLabels);});
    }
    else if(m_eStereoLabelInitType==LabelInit_LocalOptim) {
        m_oStereoLabeling.create(m_oGridSize);
        std::fill(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),(StereoLabelType)0);
        for(size_t nNodeIdx=0; nNodeIdx<nNodes; ++nNodeIdx) {
            const NodeInfo& oNode = m_vNodeInfos[nNodeIdx];
            if(oNode.nStereoVisSimUnaryFactID!=SIZE_MAX) {
                lvDbgAssert(oNode.nStereoVisSimUnaryFactID<oGM.numberOfFactors());
                lvDbgAssert(oGM.numberOfLabels(oNode.nStereoVisSimUnaryFactID)==nLabels);
                lvDbgAssert(StereoFuncID(oGM[oNode.nStereoVisSimUnaryFactID].functionIndex(),oGM[oNode.nStereoVisSimUnaryFactID].functionType())==oNode.pStereoVisSimUnaryFunc->first);
                StereoLabelType nEvalLabel = 0; // == value already assigned via std::fill
                ValueType fOptimalEnergy = oNode.pStereoVisSimUnaryFunc->second(&nEvalLabel);
                for(nEvalLabel = 1; nEvalLabel<nLabels; ++nEvalLabel) {
                    const ValueType fCurrEnergy = oNode.pStereoVisSimUnaryFunc->second(&nEvalLabel);
                    if(fOptimalEnergy>fCurrEnergy) {
                        m_oStereoLabeling(oNode.nRowIdx,oNode.nColIdx) = nEvalLabel;
                        fOptimalEnergy = fCurrEnergy;
                    }
                }
            }
        }
    }
    else /*if(m_eStereoLabelInitType==LabelInit_Explicit)*/ {
        lvAssert_(m_oStereoLabeling.isContinuous(),"internal labeling must be continuous for easier assignments");
        lvAssert_(m_oGridSize==m_oStereoLabeling.size && m_oStereoLabeling.total()==m_pStereoModel->numberOfVariables(),"internal labeling mat size & gm node count mismatch");
    }
    m_oAssocCounts = (AssocCountType)0;
    m_oAssocMap = (AssocCheckType)0;
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<(int)m_oGridSize[1]; ++nColIdx) {
            const StereoLabelType nLabel = m_oStereoLabeling(nRowIdx,nColIdx);
            const int nAssocColIdx = getOffsetCol(nColIdx,nLabel);
            if(nAssocColIdx!=INT_MAX)
                addAssoc(nRowIdx,nColIdx,nAssocColIdx);
        }
    }
    if(m_eStereoLabelOrderType==LabelOrder_Default) {
        m_vStereoLabelOrdering.resize(nLabels);
        std::iota(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),0);

    }
    else if(m_eStereoLabelOrderType==LabelOrder_Random) {
        m_vStereoLabelOrdering.resize(nLabels);
        std::iota(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),0);
        std::mt19937 oGen(m_nStereoLabelOrderRandomSeed);
        std::shuffle(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),oGen);
    }
    else /*if(m_eStereoLabelOrderType==LabelOrder_Explicit)*/ {
        lvAssert_(m_vStereoLabelOrdering.size()==nLabels,"label order array did not contain all labels");
        lvAssert_(lv::unique(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end())==lv::make_range(StereoLabelType(0),StereoLabelType(nLabels-1)),"label order array did not contain all labels");
    }
    lvAssert_(m_vStereoLabelOrdering.size()<=size_t(std::numeric_limits<StereoLabelType>::max()),"unexpected internal label count (too big for label type)");
    // @@@@@@@@@@ resegm todo
}

inline void FGStereoMatcher::GraphModelData::updateModels(const MatArrayIn& aImages) {
    for(size_t nImgIdx=0; nImgIdx<aImages.size(); ++nImgIdx)
        lvDbgAssert__(aImages[nImgIdx].dims==2 && m_oGridSize==aImages[nImgIdx].size(),"input at index=%d had the wrong size",(int)nImgIdx);
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx)
        lvDbgAssert_(aImages[nCamIdx*2+1].type()==CV_8UC1,"unexpected input mask type");
    lvAssert_(!m_bUsePrecalcFeatsNext || m_vNextFeats.size()==m_nFeatMapsPerCam*getCameraCount(),"unexpected precalculated features vec size");
    if(!m_bUsePrecalcFeatsNext)
        calcFeatures(aImages);
    else
        m_bUsePrecalcFeatsNext = false;
    static_assert(getCameraCount()==2,"lots of stuff hardcoded below for 2-cam stereo");
    const std::array<cv::Mat,2> aInputFeats = {m_vNextFeats[0],m_vNextFeats[m_nFeatMapsPerCam]};
    lvDbgAssert(!aInputFeats.empty() && lv::MatInfo(aInputFeats[0])==lv::MatInfo(aInputFeats[1]));
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    const size_t nLabels = m_vStereoLabels.size();
    lvIgnore(nLabels);
    const size_t nRealLabels = m_vStereoLabels.size()-2;
    lvDbgAssert(nLabels>3);
    lvLog(1,"Updating graph models energy terms based on input data...");
    lv::StopWatch oLocalTimer;
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::mutex oPrintMutex;
    #if USING_OPENMP
    #pragma omp parallel for collapse(2)
    #endif //USING_OPENMP
    for(int nRowIdx = 0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx = 0; nColIdx<nCols; ++nColIdx) {
            const size_t nNodeIdx = size_t(nRowIdx*nCols+nColIdx);
            lvDbgAssert(m_vNodeInfos[nNodeIdx].nRowIdx==nRowIdx && m_vNodeInfos[nNodeIdx].nColIdx==nColIdx);
            if(m_vNodeInfos[nNodeIdx].nStereoVisSimUnaryFactID!=SIZE_MAX) {
                // update stereo visual similarity unary term for each grid node
                lvDbgAssert(m_vNodeInfos[nNodeIdx].pStereoVisSimUnaryFunc);
                lvDbgAssert((&m_pStereoModel->getFunction<StereoExplicitFunction>(m_vNodeInfos[nNodeIdx].pStereoVisSimUnaryFunc->first))==(&m_vNodeInfos[nNodeIdx].pStereoVisSimUnaryFunc->second));
                StereoExplicitFunction& vUnaryFunc = m_vNodeInfos[nNodeIdx].pStereoVisSimUnaryFunc->second;
                lvDbgAssert(vUnaryFunc.dimension()==1 && vUnaryFunc.size()==nLabels);
                for(StereoLabelType nLabelIdx=0; nLabelIdx<nRealLabels; ++nLabelIdx) {
                    const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
                    const int nOffsetColIdx = nColIdx-(int)nRealLabel;
                    if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) {
                        lvDbgAssert(aImages[0].type()==CV_8UC1 && aImages[2].type()==CV_8UC1);
                        const int nColorDiff = (int)aImages[0].at<uchar>(nRowIdx,nColIdx)-(int)aImages[2].at<uchar>(nRowIdx,nOffsetColIdx);
                        vUnaryFunc(nLabelIdx) = std::min(FGSTEREOMATCH_VISSIM_COST_MAXTRUNC,ValueType(nColorDiff*nColorDiff));
                    }
                    else if(nOffsetColIdx<0)
                        vUnaryFunc(nLabelIdx) = FGSTEREOMATCH_VISSIM_COST_OOBASSOC/(-nOffsetColIdx); // reduced cost near borders
                    else //nOffsetColIdx>=nCols
                        vUnaryFunc(nLabelIdx) = FGSTEREOMATCH_VISSIM_COST_OOBASSOC/(nOffsetColIdx-nCols+1); // reduced cost near borders
                }
                vUnaryFunc(m_nStereoDontCareLabelIdx) = ValueType(100000); // @@@@ check roi, if dc set to 0, otherwise set to inf
                vUnaryFunc(m_nStereoOccludedLabelIdx) = ValueType(100000);//FGSTEREOMATCH_VISSIM_COST_OCCLUDED;
            }
            // update stereo smoothness pairwise terms for each grid node
            for(size_t nOrientIdx=0; nOrientIdx<m_vNodeInfos[nNodeIdx].anStereoSmoothPairwFactIDs.size(); ++nOrientIdx) {
                if(m_vNodeInfos[nNodeIdx].anStereoSmoothPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                    lvDbgAssert(m_vNodeInfos[nNodeIdx].apStereoSmoothPairwFuncs[nOrientIdx]);
                    lvDbgAssert((&m_pStereoModel->getFunction<StereoExplicitFunction>(m_vNodeInfos[nNodeIdx].apStereoSmoothPairwFuncs[nOrientIdx]->first))==(&m_vNodeInfos[nNodeIdx].apStereoSmoothPairwFuncs[nOrientIdx]->second));
                    StereoExplicitFunction& vPairwiseFunc = m_vNodeInfos[nNodeIdx].apStereoSmoothPairwFuncs[nOrientIdx]->second;
                    lvDbgAssert(vPairwiseFunc.dimension()==2 && vPairwiseFunc.size()==nLabels*nLabels);
                    for(StereoLabelType nLabelIdx1=0; nLabelIdx1<nRealLabels; ++nLabelIdx1) {
                        for(StereoLabelType nLabelIdx2=0; nLabelIdx2<nRealLabels; ++nLabelIdx2) {
                            const OutputLabelType nRealLabel1 = getRealLabel(nLabelIdx1);
                            const OutputLabelType nRealLabel2 = getRealLabel(nLabelIdx2);
                            const int nRealLabelDiff = (int)nRealLabel1-(int)nRealLabel2;
                            vPairwiseFunc(nLabelIdx1,nLabelIdx2) = std::min(FGSTEREOMATCH_LBLSIM_COST_MAXTRUNC,ValueType(nRealLabelDiff*nRealLabelDiff));
                        }
                    }
                    for(size_t nLabelIdx=0; nLabelIdx<nRealLabels; ++nLabelIdx) {
                        // @@@ change later for vis-data-dependent or roi-dependent energies?
                        vPairwiseFunc(m_nStereoDontCareLabelIdx,nLabelIdx) = ValueType(100000);
                        vPairwiseFunc(m_nStereoOccludedLabelIdx,nLabelIdx) = ValueType(100000); // @@@@ FGSTEREOMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                        vPairwiseFunc(nLabelIdx,m_nStereoDontCareLabelIdx) = ValueType(100000);
                        vPairwiseFunc(nLabelIdx,m_nStereoOccludedLabelIdx) = ValueType(100000); // @@@@ FGSTEREOMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                    }
                    vPairwiseFunc(m_nStereoDontCareLabelIdx,m_nStereoDontCareLabelIdx) = ValueType(0);
                    vPairwiseFunc(m_nStereoOccludedLabelIdx,m_nStereoOccludedLabelIdx) = ValueType(0);
                }
            }
            ++nProcessedNodeCount;
            if(nColIdx==0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tupdate:",float(nProcessedNodeCount)/size_t(nRows*nCols));
            }
        }
    }
    lv::cleanConsoleRow();
    lvLog_(1,"Graph energy terms update completed in %f second(s).",oLocalTimer.tock());
    resetLabelings();
    m_bModelUpToDate = true;
}

inline void FGStereoMatcher::GraphModelData::calcFeatures(const MatArrayIn& aImages, cv::Mat* pFeatsPacket) {
    m_vNextFeats.resize(m_nFeatMapsPerCam*getCameraCount());
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvAssert__(aImages[nCamIdx*2].dims==2 && m_oGridSize==aImages[nCamIdx*2].size(),"input image in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert__(aImages[nCamIdx*2+1].dims==2 && m_oGridSize==aImages[nCamIdx*2+1].size(),"input mask in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert_(aImages[nCamIdx*2].type()==CV_8UC1 || aImages[nCamIdx*2].type()==CV_8UC3,"unexpected input image type");
        lvAssert_(aImages[nCamIdx*2+1].type()==CV_8UC1,"unexpected input mask type");
#if FGSTEREOMATCH_CONFIG_USE_DASC_FEATS
        dynamic_cast<DASC&>(*m_pFeatExtractor).compute2(aImages[nCamIdx*2],m_vNextFeats[nCamIdx*m_nFeatMapsPerCam]);
#elif FGSTEREOMATCH_CONFIG_USE_LSS_FEATS
        @@@
#endif //FGSTEREOMATCH_CONFIG_USE_..._FEATS
        cv::Mat oBlurredInput;
        cv::GaussianBlur(aImages[nCamIdx*2],oBlurredInput,cv::Size(3,3),0);
        cv::Mat oBlurredGrayInput;
        if(oBlurredInput.channels()==3)
            cv::cvtColor(oBlurredInput,oBlurredGrayInput,cv::COLOR_BGR2GRAY);
        else
            oBlurredGrayInput = oBlurredInput;
        cv::Mat oGradInput_X,oGradInput_Y;
        cv::Scharr(oBlurredGrayInput,oGradInput_X,CV_16S,1,0);
        cv::Scharr(oBlurredGrayInput,oGradInput_Y,CV_16S,0,1);
        cv::Mat oAbsGradInput_X,oAbsGradInput_Y;
        cv::convertScaleAbs(oGradInput_X,oAbsGradInput_X);
        cv::convertScaleAbs(oGradInput_Y,oAbsGradInput_Y);
        cv::addWeighted(oAbsGradInput_X,0.5,oAbsGradInput_Y,0.5,0,m_vNextFeats[nCamIdx*m_nFeatMapsPerCam+2]);
        cv::distanceTransform(aImages[nCamIdx*2+1]==0,m_vNextFeats[nCamIdx*m_nFeatMapsPerCam+1],cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
    }
    std::vector<lv::MatInfo> vLastPackInfo = m_vFeatPackInfo;
    if(pFeatsPacket)
        *pFeatsPacket = lv::packData(m_vNextFeats,&m_vFeatPackInfo);
    else {
        m_vFeatPackInfo.resize(m_vNextFeats.size());
        for(size_t nFeatMapIdx=0; nFeatMapIdx<m_vNextFeats.size(); ++nFeatMapIdx)
            m_vFeatPackInfo[nFeatMapIdx] = lv::MatInfo(m_vNextFeats[nFeatMapIdx]);
    }
    lvAssert_(vLastPackInfo.empty() || m_vFeatPackInfo==vLastPackInfo,"packed features info mismatch (should stay constant for all inputs)");
    m_bModelUpToDate = false;
}

inline void FGStereoMatcher::GraphModelData::setNextFeatures(const cv::Mat& oPackedFeats) {
    lvAssert_(!oPackedFeats.empty(),"features packet must be non-empty");
    std::vector<lv::MatInfo> vExpectedPackInfo;
#if FGSTEREOMATCH_CONFIG_USE_DASC_FEATS
    const lv::MatInfo oFeatsInfo = dynamic_cast<DASC&>(*m_pFeatExtractor).getOutputInfo(lv::MatInfo(m_oGridSize,CV_8UC3)); // @@@@ assume type wont change output info
#elif FGSTEREOMATCH_CONFIG_USE_LSS_FEATS
    const lv::MatInfo oFeatsInfo = dynamic_cast<LSS&>(*m_pFeatExtractor).getOutputInfo(lv::MatInfo(m_oGridSize,CV_8UC3)); // @@@@ assume type wont change output info
#endif //FGSTEREOMATCH_CONFIG_USE_..._FEATS
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        vExpectedPackInfo.push_back(oFeatsInfo); // input features map
        vExpectedPackInfo.push_back(lv::MatInfo(m_oGridSize,CV_32FC1)); // dist transf map
        vExpectedPackInfo.push_back(lv::MatInfo(m_oGridSize,CV_8UC1)); // grad mag map
    }
    lvDbgAssert((vExpectedPackInfo.size()%m_nFeatMapsPerCam)==0);
    lvDbgAssert_(m_vFeatPackInfo.empty() || m_vFeatPackInfo==vExpectedPackInfo,"packed features info mismatch (unexpected hardcoded type)");
    m_oNextPackedFeats = oPackedFeats; // get pointer to data instead of copy; provider should not overwrite until next 'apply' call!
    m_vNextFeats = lv::unpackData(m_oNextPackedFeats,vExpectedPackInfo);
    m_bUsePrecalcFeatsNext = true;
    m_bModelUpToDate = false;
}

inline FGStereoMatcher::OutputLabelType FGStereoMatcher::GraphModelData::getMinOffset() const {
    return m_vStereoLabels[0];
}

inline FGStereoMatcher::OutputLabelType FGStereoMatcher::GraphModelData::getMaxOffset() const {
    lvDbgAssert(m_vStereoLabels.size()>3);
    return m_vStereoLabels[m_vStereoLabels.size()-3];
}

inline FGStereoMatcher::OutputLabelType FGStereoMatcher::GraphModelData::getRealLabel(StereoLabelType nLabel) const {
    lvDbgAssert(nLabel<m_vStereoLabels.size());
    return m_vStereoLabels[nLabel];
}

inline FGStereoMatcher::StereoLabelType FGStereoMatcher::GraphModelData::getInternalLabel(OutputLabelType nRealLabel) const {
    lvDbgAssert(nRealLabel==s_nStereoOccludedLabel || nRealLabel==s_nStereoDontCareLabel || (nRealLabel>=getMinOffset() && nRealLabel<=getMaxOffset()));
    return (StereoLabelType)std::distance(m_vStereoLabels.begin(),std::find(m_vStereoLabels.begin(),m_vStereoLabels.end(),nRealLabel));
}

inline int FGStereoMatcher::GraphModelData::getOffsetCol(int nColIdx, StereoLabelType nLabel) const {
    if(nLabel>=m_nStereoDontCareLabelIdx) // both special labels treated here
        return INT_MAX; // return bad index
    const OutputLabelType nRealLabel = getRealLabel(nLabel);
    const int nOffsetColIdx = nColIdx-(int)nRealLabel;
    return (nOffsetColIdx>=0 && nOffsetColIdx<int(m_oGridSize[1]))?nOffsetColIdx:INT_MAX;
}

inline size_t FGStereoMatcher::GraphModelData::getOffsetNode(size_t nNodeIdx, StereoLabelType nLabel) const {
    if(nLabel>=m_nStereoDontCareLabelIdx) // both special labels treated here
        return SIZE_MAX; // return bad index
    const OutputLabelType nRealLabel = getRealLabel(nLabel);
    const int nColIdx = int(nNodeIdx%m_oGridSize[1]);
    const int nOffsetColIdx = nColIdx-(int)nRealLabel;
    return (nOffsetColIdx>=0 && nOffsetColIdx<int(m_oGridSize[1]))?(nNodeIdx-size_t(nRealLabel)):SIZE_MAX;
}

inline FGStereoMatcher::AssocCountType FGStereoMatcher::GraphModelData::getAssocCount(int nRowIdx, int nColIdx) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0]);
    lvDbgAssert(nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    return ((AssocCountType*)m_oAssocCounts.data)[(nRowIdx/m_nStereoLabelStep)*m_oAssocCounts.cols + nColIdx/m_nStereoLabelStep];
}

inline void FGStereoMatcher::GraphModelData::addAssoc(int nRowIdx, int nColIdx, int nAssocColIdx) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0]);
    lvDbgAssert(nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(nAssocColIdx>=0 && nAssocColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    AssocCheckType& nAssocCheck = ((AssocCheckType*)m_oAssocMap.data)[((nRowIdx/m_nStereoLabelStep)*m_oAssocMap.size[1] + nAssocColIdx/m_nStereoLabelStep)*m_oAssocMap.size[2] + nColIdx/m_nStereoLabelStep];
    lvDbgAssert(nAssocCheck==AssocCheckType(0));
    nAssocCheck = AssocCheckType(-1);
    AssocCountType& nAssocCount = ((AssocCountType*)m_oAssocCounts.data)[(nRowIdx/m_nStereoLabelStep)*m_oAssocCounts.cols + nAssocColIdx/m_nStereoLabelStep];
    lvDbgAssert((&nAssocCount)==&m_oAssocCounts(int(nRowIdx/m_nStereoLabelStep),int(nAssocColIdx/m_nStereoLabelStep)));
    lvDbgAssert(nAssocCount<std::numeric_limits<AssocCountType>::max());
    ++nAssocCount;
}

inline void FGStereoMatcher::GraphModelData::removeAssoc(int nRowIdx, int nColIdx, int nAssocColIdx) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0]);
    lvDbgAssert(nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(nAssocColIdx>=0 && nAssocColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    AssocCheckType& nAssocCheck = ((AssocCheckType*)m_oAssocMap.data)[((nRowIdx/m_nStereoLabelStep)*m_oAssocMap.size[1] + nAssocColIdx/m_nStereoLabelStep)*m_oAssocMap.size[2] + nColIdx/m_nStereoLabelStep];
    lvDbgAssert(nAssocCheck==AssocCheckType(-1));
    nAssocCheck = AssocCheckType(0);
    AssocCountType& nAssocCount = ((AssocCountType*)m_oAssocCounts.data)[(nRowIdx/m_nStereoLabelStep)*m_oAssocCounts.cols + nAssocColIdx/m_nStereoLabelStep];
    lvDbgAssert((&nAssocCount)==&m_oAssocCounts(int(nRowIdx/m_nStereoLabelStep),int(nAssocColIdx/m_nStereoLabelStep)));
    lvDbgAssert(nAssocCount>AssocCountType(0));
    --nAssocCount;
}

inline FGStereoMatcher::ValueType FGStereoMatcher::GraphModelData::getTotalAssocEnergy() const {
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
    const size_t nTotNodeCount = m_oGridSize.total();
    ValueType tEnergy = ValueType(0);
    for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
        const int nAssocCount = (int)getAssocCount(m_vNodeInfos[nNodeIdx].nRowIdx,m_vNodeInfos[nNodeIdx].nColIdx);
        tEnergy += ValueType(float(nAssocCount*nAssocCount)*FGSTEREOMATCH_UNIQUE_COST_OVERASSOC/m_nStereoLabelStep);
        const StereoLabelType nCurrLabel = ((StereoLabelType*)m_oStereoLabeling.data)[nNodeIdx];
        if(nCurrLabel>=m_nStereoDontCareLabelIdx) // both special labels treated here
            tEnergy += ValueType(100000); // @@@@ dirty
    }
    return tEnergy;
}

inline FGStereoMatcher::ValueType FGStereoMatcher::GraphModelData::getAssocEnergyDiff(size_t nNodeIdx, StereoLabelType nCurrLabel, StereoLabelType nNewLabel) const {
    lvDbgAssert(nNodeIdx<m_vNodeInfos.size());
    if(nCurrLabel==nNewLabel)
        return ValueType(0);
    ValueType tEnergyDiff = ValueType(0);
    if(nCurrLabel<m_nStereoDontCareLabelIdx) {
        const OutputLabelType nCurrRealLabel = getRealLabel(nCurrLabel);
        const int nCurrOffsetColIdx = m_vNodeInfos[nNodeIdx].nColIdx-(int)nCurrRealLabel;
        if(nCurrOffsetColIdx>=0 && nCurrOffsetColIdx<int(m_oGridSize[1])) {
            const int nOrigAssocCount = (int)getAssocCount(m_vNodeInfos[nNodeIdx].nRowIdx,nCurrOffsetColIdx);
            lvDbgAssert(nOrigAssocCount>0); // cannot be zero, must have an association at current label
            const int nFutureAssocCount = nOrigAssocCount-1;
            tEnergyDiff -= ValueType(float((nOrigAssocCount*nOrigAssocCount)-(nFutureAssocCount*nFutureAssocCount))*FGSTEREOMATCH_UNIQUE_COST_OVERASSOC/m_nStereoLabelStep);
        }
        else if(nCurrOffsetColIdx<0)
            tEnergyDiff -= FGSTEREOMATCH_UNIQUE_COST_OOBASSOC/(-nCurrOffsetColIdx); // provide reduced cost near borders
        else //nCurrOffsetColIdx>=int(m_oGridSize[1])
            tEnergyDiff -= FGSTEREOMATCH_UNIQUE_COST_OOBASSOC/(nCurrOffsetColIdx-int(m_oGridSize[1])+1); // provide reduced cost near borders
    }
    else
        tEnergyDiff -= ValueType(100000); // @@@@ dirty
    //std::cout << "n=" << nNodeIdx << ", labels=" << (int)nCurrLabel << "," << (int)nNewLabel << "    e=" << tEnergyDiff << std::endl;
    if(nNewLabel>=m_nStereoDontCareLabelIdx) // both special labels treated here
        return tEnergyDiff+ValueType(100000); // @@@@ dirty
    const OutputLabelType nNewRealLabel = getRealLabel(nNewLabel);
    const int nNewOffsetColIdx = m_vNodeInfos[nNodeIdx].nColIdx-(int)nNewRealLabel;
    if(nNewOffsetColIdx<0)
        return tEnergyDiff+FGSTEREOMATCH_UNIQUE_COST_OOBASSOC/(-nNewOffsetColIdx); // provide reduced cost near borders
    else //nNewOffsetColIdx>=int(m_oGridSize[1])
        return tEnergyDiff+FGSTEREOMATCH_UNIQUE_COST_OOBASSOC/(nNewOffsetColIdx-int(m_oGridSize[1])+1); // provide reduced cost near borders
    const int nOrigAssocCount = getAssocCount(m_vNodeInfos[nNodeIdx].nRowIdx,nNewOffsetColIdx);
    const int nFutureAssocCount = nOrigAssocCount+1;
    return tEnergyDiff+ValueType(float((nFutureAssocCount*nFutureAssocCount)-(nOrigAssocCount*nOrigAssocCount))*FGSTEREOMATCH_UNIQUE_COST_OVERASSOC/m_nStereoLabelStep);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline FGStereoMatcher::StereoGraphInference::StereoGraphInference(const GraphModelData& oData) :
        m_oData(oData) {
    lvDbgAssert_(m_oData.m_pStereoModel && m_oData.m_pStereoModel->numberOfFactors()>0,"invalid graph");
    const StereoModelType& oGM = *m_oData.m_pStereoModel;
    for(size_t nFactIdx=0; nFactIdx<m_oData.m_pStereoModel->numberOfFactors(); ++nFactIdx)
        lvDbgAssert__(oGM[nFactIdx].numberOfVariables()<=s_nMaxOrder,"graph had some factors of order %d (max allowed is %d)",(int)oGM[nFactIdx].numberOfVariables(),(int)s_nMaxOrder);
    lvDbgAssert_(m_oData.m_pStereoModel->numberOfVariables()>0 && m_oData.m_pStereoModel->numberOfVariables()==(IndexType)m_oData.m_oGridSize.total(),"graph node count must match grid size");
    for(size_t nNodeIdx=0; nNodeIdx<m_oData.m_pStereoModel->numberOfVariables(); ++nNodeIdx)
        lvDbgAssert_(m_oData.m_pStereoModel->numberOfLabels(nNodeIdx)==m_oData.m_vStereoLabels.size(),"graph nodes must all have the same number of labels");
    lvIgnore(oGM);
}

inline std::string FGStereoMatcher::StereoGraphInference::name() const {
    return std::string("litiv-stereo-matcher");
}

inline const FGStereoMatcher::StereoModelType& FGStereoMatcher::StereoGraphInference::graphicalModel() const {
    return *m_oData.m_pStereoModel;
}

inline opengm::InferenceTermination FGStereoMatcher::StereoGraphInference::infer() {
    EmptyVisitorType visitor;
    return infer(visitor);
}

inline void FGStereoMatcher::StereoGraphInference::setStartingPoint(typename std::vector<StereoLabelType>::const_iterator begin) {
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_pStereoModel->numberOfVariables()),lv::make_range(StereoLabelType(0),StereoLabelType(m_oData.m_vStereoLabels.size()-1))).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_oStereoLabeling.isContinuous() && m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_pStereoModel->numberOfVariables(),m_oData.m_oStereoLabeling.begin());
}

inline void FGStereoMatcher::StereoGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),m_oData.m_vStereoLabels).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_oStereoLabeling.begin(),[&](const OutputLabelType& nRealLabel){return m_oData.getInternalLabel(nRealLabel);});
}

inline opengm::InferenceTermination FGStereoMatcher::StereoGraphInference::arg(std::vector<StereoLabelType>& oLabeling, const size_t n) const {
    if(n==1) {
        lvDbgAssert_(m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_oStereoLabeling.total());
        std::copy(m_oData.m_oStereoLabeling.begin(),m_oData.m_oStereoLabeling.end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones!
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

inline void FGStereoMatcher::StereoGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgAssert_(m_oData.m_oStereoLabeling.isContinuous() && m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_oStereoLabeling.size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_oStereoLabeling.begin(),m_oData.m_oStereoLabeling.end(),oLabeling.begin(),[&](const StereoLabelType& nLabel){return m_oData.getRealLabel(nLabel);});
}

inline FGStereoMatcher::ValueType FGStereoMatcher::StereoGraphInference::value() const {
    lvDbgAssert_(m_oData.m_pStereoModel->numberOfVariables()==m_oData.m_oStereoLabeling.total(),"graph node count and labeling mat size mismatch");
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_oStereoLabeling.size,"output labeling must be a 2d grid");
    lvAssert_(m_oData.m_bModelUpToDate,"model must be up-to-date with new data before inference can begin");
    return m_oData.m_pStereoModel->evaluate((StereoLabelType*)m_oData.m_oStereoLabeling.data)+m_oData.getTotalAssocEnergy();
}

template<typename TVisitor> inline
opengm::InferenceTermination FGStereoMatcher::StereoGraphInference::infer(TVisitor& oVisitor) {
    lvDbgAssert_(m_oData.m_pStereoModel->numberOfVariables()==m_oData.m_oStereoLabeling.total(),"graph node count and labeling mat size mismatch");
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_oStereoLabeling.size,"output labeling must be a 2d grid");
    lvAssert_(m_oData.m_bModelUpToDate,"model must be up-to-date with new data before inference can begin");
    // @@@@@@ use one gm labeling/output to infer the stereo result for another camera?
    const StereoModelType& oGM = *m_oData.m_pStereoModel;
    const size_t nTotNodeCount = oGM.numberOfVariables();
    lvDbgAssert(nTotNodeCount==m_oData.m_oGridSize.total());
    lvDbgAssert(nTotNodeCount==m_oData.m_vNodeInfos.size());
    const size_t nLabels = m_oData.m_vStereoLabels.size();
    const lv::MatSize& oGridSize = m_oData.m_oGridSize;
    const size_t nGridBorderSize = m_oData.m_nGridBorderSize;
    const cv::Rect oFeatROI((int)nGridBorderSize,(int)nGridBorderSize,(int)(oGridSize[1]-nGridBorderSize*2),(int)(oGridSize[0]-nGridBorderSize*2));
    // @@@@@@ use member-alloc QPBO here instead of stack
    kolmogorov::qpbo::QPBO<ValueType> oBinaryEnergyMinimizer((int)nTotNodeCount,0); // @@@@@@@ preset max edge count using max clique size times node count
    // @@@@@@ THIS IS A DYNAMIC MRF OPTIMIZATION PROBLEM; MUST TRY TO REUSE MRF STRUCTURE
    std::array<ValueType,s_nMaxCliqueAssign> aCliqueCoeffs;
    std::array<StereoLabelType,s_nMaxOrder> aCliqueLabels;
    std::array<typename HigherOrderEnergy<ValueType,s_nMaxOrder>::VarId,s_nMaxOrder> aTermEnergyLUT;
    cv::Mat_<StereoLabelType>& oLabeling = m_oData.m_oStereoLabeling;
    size_t nMoveIter = 0, nConsecUnchangedLabels = 0, nOrderingIdx = 0;
    lvDbgAssert(m_oData.m_vStereoLabelOrdering.size()==m_oData.m_vStereoLabels.size());
    StereoLabelType nAlphaLabel = m_oData.m_vStereoLabelOrdering[nOrderingIdx];
    lv::StopWatch oLocalTimer;
    oVisitor.begin(*this);
    // each iter below is an alpha-exp move based on A. Fix's primal-dual energy minimization method for higher-order MRFs
    // see "A Primal-Dual Algorithm for Higher-Order Multilabel Markov Random Fields" in CVPR2014 for more info (doi = 10.1109/CVPR.2014.149)
    while(++nMoveIter<=m_oData.m_nMaxIterCount && nConsecUnchangedLabels<nLabels) {
        if(lv::getVerbosity()>=3) {
            /*if(m_oData.m_oGridSize.total()>=100) {
                const int nTestRowIdx = 115;
                const int nTestColMin = 50, nTestColMax = 110;
                std::cout << " assoc @ row#" << nTestRowIdx << " = ";
                for(int nColIdx=nTestColMin; nColIdx<nTestColMax; ++nColIdx) {
                    std::cout << (int)m_oData.getAssocCount(nTestRowIdx,nColIdx) << ", ";
                }
                std::cout << std::endl;
                std::cout << " disp @ row#" << nTestRowIdx << " = ";
                for(int nColIdx=nTestColMin; nColIdx<nTestColMax; ++nColIdx) {
                    std::cout << (int)m_oData.getRealLabel(m_oData.m_oStereoLabeling(nTestRowIdx,nColIdx)) << ", ";
                }
                std::cout << std::endl;
            }
            else {
                std::cout << "assoc count (tot=" << (size_t)cv::sum(m_oData.m_oAssocCounts)[0] << ") = " << lv::to_string(m_oData.m_oAssocCounts) << std::endl;
                std::cout << "disp = " << lv::to_string(m_oData.m_oStereoLabeling) << std::endl;
            }*/
            std::cout << "assoc count = " << lv::to_string(m_oData.m_oAssocCounts(oFeatROI)) << std::endl;
            std::cout << "disp = " << lv::to_string(m_oData.m_oStereoLabeling(oFeatROI)) << std::endl;
            cv::Mat oCurrAssocCountsDisplay = FGStereoMatcher::getAssocCountsMapDisplay(m_oData);
            cv::resize(oCurrAssocCountsDisplay,oCurrAssocCountsDisplay,cv::Size(640,480),0,0,cv::INTER_NEAREST);
            cv::imshow("assoc",oCurrAssocCountsDisplay);
            cv::Mat oCurrLabelingDisplay = FGStereoMatcher::getStereoMapDisplay(m_oData);
            cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(640,480),0,0,cv::INTER_NEAREST);
            cv::imshow("stereo",oCurrLabelingDisplay);
            cv::waitKey(0);
            std::cout << "@@@ next label idx = " << (int)nAlphaLabel << ",  real = " << (int)m_oData.getRealLabel(nAlphaLabel) << std::endl;
        }
        HigherOrderEnergy<ValueType,s_nMaxOrder> oHigherOrderEnergyReducer; // @@@@@@ check if HigherOrderEnergy::Clear() can be used instead of realloc
        oHigherOrderEnergyReducer.AddVars((int)nTotNodeCount);
        const auto lHigherOrderFactorAdder = [&](size_t nFactIdx, size_t nCurrFactOrder) {
            lvDbgAssert(oGM[nFactIdx].numberOfVariables()==nCurrFactOrder);
            const size_t nAssignCount = 1UL<<nCurrFactOrder;
            std::fill_n(aCliqueCoeffs.begin(),nAssignCount,(ValueType)0);
            for(size_t nAssignIdx=0; nAssignIdx<nAssignCount; ++nAssignIdx) {
                for(size_t nVarIdx=0; nVarIdx<nCurrFactOrder; ++nVarIdx)
                    aCliqueLabels[nVarIdx] = (nAssignIdx&(1<<nVarIdx))?nAlphaLabel:oLabeling((int)oGM[nFactIdx].variableIndex(nVarIdx));
                for(size_t nAssignSubsetIdx=1; nAssignSubsetIdx<nAssignCount; ++nAssignSubsetIdx) {
                    if(!(nAssignIdx&~nAssignSubsetIdx)) {
                        int nParityBit = 0;
                        for(size_t nVarIdx=0; nVarIdx<nCurrFactOrder; ++nVarIdx)
                            nParityBit ^= (((nAssignIdx^nAssignSubsetIdx)&(1<<nVarIdx))!=0);
                        const ValueType fCurrAssignEnergy = oGM[nFactIdx](aCliqueLabels.begin());
                        aCliqueCoeffs[nAssignSubsetIdx] += nParityBit?-fCurrAssignEnergy:fCurrAssignEnergy;
                    }
                }
            }
            for(size_t nAssignSubsetIdx=1; nAssignSubsetIdx<nAssignCount; ++nAssignSubsetIdx) {
                int nCurrTermDegree = 0;
                for(size_t nVarIdx=0; nVarIdx<nCurrFactOrder; ++nVarIdx)
                    if(nAssignSubsetIdx&(1<<nVarIdx))
                        aTermEnergyLUT[nCurrTermDegree++] = (typename HigherOrderEnergy<ValueType,s_nMaxOrder>::VarId)oGM[nFactIdx].variableIndex(nVarIdx);
                std::sort(aTermEnergyLUT.begin(),aTermEnergyLUT.begin()+nCurrTermDegree);
                oHigherOrderEnergyReducer.AddTerm(aCliqueCoeffs[nAssignSubsetIdx],nCurrTermDegree,aTermEnergyLUT.data());
            }
        };
        ValueType tMinEnergy = ValueType(0);
        cv::Mat_<ValueType> UNARIES_VISSIM(m_oData.m_oGridSize(),ValueType(0));
        cv::Mat_<ValueType> UNARIES_ASSOC(m_oData.m_oGridSize(),ValueType(0));
        for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
            const GraphModelData::NodeInfo& oNode = m_oData.m_vNodeInfos[nNodeIdx];
            // manually add 1st order factors while evaluating new assoc energy
            if(oNode.nStereoVisSimUnaryFactID!=SIZE_MAX) {
                const StereoLabelType& nInitLabel = ((StereoLabelType*)oLabeling.data)[nNodeIdx];
                lvDbgAssert(&nInitLabel==&oLabeling(oNode.nRowIdx,oNode.nColIdx));
                const float fAssocScaleFact = float(m_oData.getAssocCount(oNode.nRowIdx,oNode.nColIdx)+1);
                const ValueType tVisSimEnergyInit = ValueType(oNode.pStereoVisSimUnaryFunc->second(&nInitLabel)/fAssocScaleFact);
                const ValueType tVisSimEnergyModif = ValueType(oNode.pStereoVisSimUnaryFunc->second(&nAlphaLabel)/fAssocScaleFact);
                const ValueType tVisSimEnergyDiff = tVisSimEnergyModif-tVisSimEnergyInit;
                const ValueType tAssocEnergyDiff = m_oData.getAssocEnergyDiff(nNodeIdx,nInitLabel,nAlphaLabel);
                oHigherOrderEnergyReducer.AddUnaryTerm((int)nNodeIdx,tVisSimEnergyDiff+tAssocEnergyDiff);
                UNARIES_VISSIM(m_oData.m_vNodeInfos[nNodeIdx].nRowIdx,m_oData.m_vNodeInfos[nNodeIdx].nColIdx) = tVisSimEnergyDiff;
                UNARIES_ASSOC(m_oData.m_vNodeInfos[nNodeIdx].nRowIdx,m_oData.m_vNodeInfos[nNodeIdx].nColIdx) = tAssocEnergyDiff;
            }
            // now add 2nd order & higher order factors via lambda
            if(oNode.anStereoSmoothPairwFactIDs[0]!=SIZE_MAX)
                lHigherOrderFactorAdder(oNode.anStereoSmoothPairwFactIDs[0],2);
            if(oNode.anStereoSmoothPairwFactIDs[1]!=SIZE_MAX)
                lHigherOrderFactorAdder(oNode.anStereoSmoothPairwFactIDs[1],2);
        }
        std::cout << "unaries_vissim = " << lv::to_string(UNARIES_VISSIM(oFeatROI)) << std::endl;
        std::cout << "unaries_assoc = " << lv::to_string(UNARIES_ASSOC(oFeatROI)) << std::endl;
        oBinaryEnergyMinimizer.Reset();
        oHigherOrderEnergyReducer.ToQuadratic(oBinaryEnergyMinimizer); // @@@@@ internally might call addNodes/edges to QPBO object; optim tbd
        //@@@@@@ oBinaryEnergyMinimizer.AddPairwiseTerm(nodei,nodej,val00,val01,val10,val11); @@@ can still add more if needed
        oBinaryEnergyMinimizer.Solve();
        oBinaryEnergyMinimizer.ComputeWeakPersistencies();
        size_t nChangedLabelings = 0;
        cv::Mat_<uchar> SWAPS(m_oData.m_oGridSize(),0);
        for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
            const int nMoveLabel = oBinaryEnergyMinimizer.GetLabel((int)nNodeIdx);
            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
            if(nMoveLabel==1) { // node label changed to alpha
                SWAPS(m_oData.m_vNodeInfos[nNodeIdx].nRowIdx,m_oData.m_vNodeInfos[nNodeIdx].nColIdx) = 1;
                const std::div_t oDivRes = std::div((int)nNodeIdx,(int)oGridSize[1]);
                const StereoLabelType nOldLabel = oLabeling(oDivRes.quot,oDivRes.rem);
                const int nOldAssocColIdx = m_oData.getOffsetCol(oDivRes.rem,nOldLabel);
                if(nOldAssocColIdx!=INT_MAX)
                    m_oData.removeAssoc(oDivRes.quot,oDivRes.rem,nOldAssocColIdx);
                oLabeling(oDivRes.quot,oDivRes.rem) = nAlphaLabel;
                const int nNewAssocColIdx = m_oData.getOffsetCol(oDivRes.rem,nAlphaLabel);
                if(nNewAssocColIdx!=INT_MAX)
                    m_oData.addAssoc(oDivRes.quot,oDivRes.rem,nNewAssocColIdx);
                ++nChangedLabelings;
            }
        }
        std::cout << "swaps (tot=" << (size_t)cv::sum(SWAPS(oFeatROI))[0] << ") = " << lv::to_string(SWAPS(oFeatROI)) << std::endl;
        nConsecUnchangedLabels = (nChangedLabelings>0)?0:nConsecUnchangedLabels+1;
        // @@@@ order of future moves can be influenced by labels that cause the most changes? (but only late, to avoid bad local minima?)
        nAlphaLabel = m_oData.m_vStereoLabelOrdering[(++nOrderingIdx%=nLabels)];
        if(oVisitor(*this)!=opengm::visitors::VisitorReturnFlag::ContinueInf)
            break;
    }
    oVisitor.end(*this);
    lvLog_(1,"Inference completed in %f second(s).",oLocalTimer.tock());
    return opengm::InferenceTermination::NORMAL;
}

inline cv::Mat FGStereoMatcher::getStereoMapDisplay(const GraphModelData& oData) {
    lvAssert(!oData.m_oStereoLabeling.empty() && oData.m_oGridSize==oData.m_oStereoLabeling.size);
    const OutputLabelType nMinDispLabel = oData.getMinOffset();
    const OutputLabelType nMaxDispLabel = oData.getMaxOffset();
    const float fRescaleFact = float(UCHAR_MAX)/(int(nMaxDispLabel)-nMinDispLabel+1);
    cv::Mat oOutput(oData.m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<oData.m_oStereoLabeling.rows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<oData.m_oStereoLabeling.cols; ++nColIdx) {
            const OutputLabelType nRealLabel = oData.getRealLabel(oData.m_oStereoLabeling(nRowIdx,nColIdx));;
            if(nRealLabel==s_nStereoDontCareLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,128,128);
            else if(nRealLabel==s_nStereoOccludedLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(128,0,128);
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b::all(uchar((nRealLabel-nMinDispLabel)*fRescaleFact));
        }
    }
    return oOutput;
}

inline cv::Mat FGStereoMatcher::getAssocCountsMapDisplay(const GraphModelData& oData) {
    lvAssert(!oData.m_oAssocCounts.empty() && oData.m_oGridSize==oData.m_oAssocCounts.size);
    double dMax;
    cv::minMaxIdx(oData.m_oAssocCounts,nullptr,&dMax);
    const float fRescaleFact = float(UCHAR_MAX)/(int(dMax)+1);
    cv::Mat oOutput(oData.m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<(int)oData.m_oGridSize[0]; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<(int)oData.m_oGridSize[1]; ++nColIdx) {
            const AssocCountType nCount = oData.getAssocCount(nRowIdx,nColIdx);
            oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b::all(uchar(nCount*fRescaleFact));
        }
    }
    return oOutput;
}