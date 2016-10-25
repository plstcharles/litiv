
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

#include "litiv/imgproc/CosegmentationUtils.hpp"
#include "litiv/utils/opengm.hpp"

/// defines the internal type to be used for label values in the graph
#define STEREOMATCH_LABEL_TYPE        size_t
/// defines the max factor order expected in the graphical model
#define STEREOMATCH_MAX_FACT_ORDER    (6)
/// defines whether to use a global potts function for the label smoothness term, or an explicit one
#define STEREOMATCH_USE_SMOOTH_POTTS  1

// unary costs params
#define STEREOMATCH_VISSIM_COST_OCCLUDED     10.0f
#define STEREOMATCH_VISSIM_COST_MAXTRUNC     30.0f
#define STEREOMATCH_UNIQUE_COST_OVERASSOC    10.0f
#define STEREOMATCH_UNIQUE_COST_OOBASSOC     10.0f
// pairwise costs params
#define STEREOMATCH_LBLSIM_COST_EQUAL         0.0f
#define STEREOMATCH_LBLSIM_COST_NEQUAL        10.0f
// higher order costs params
// ...

/// this stereo matcher assumes input images are rectified and have the same size
struct StereoMatcher : public ICosegmentor<STEREOMATCH_LABEL_TYPE,2> {

    using ValueType = float; // type used for factor values (@@@@ could be integer? retest speed later?)
    using IndexType = size_t; // type used for node indexing (note: pretty much hardcoded everywhere in impl below)
    using OpType = opengm::Adder; // operation used to combine factors
    using ExplicitFunction = opengm::ExplicitFunction<ValueType,IndexType,LabelType>; // shortcut for explicit function
    using PottsFunction = opengm::PottsFunction<ValueType,IndexType,LabelType>; // shortcut for Potts function
    using FunctionTypeList = opengm::meta::TypeListGenerator<ExplicitFunction,PottsFunction>::type;  // list of all function the model can use
    using SpaceType = opengm::SimpleDiscreteSpace<IndexType,LabelType>; // shortcut for discrete space type (simple = all nodes have the same # of labels)
    using AccumulatorType = opengm::Minimizer; // shortcut for energy accumulator type
    using ModelType = opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>; // shortcut for graphical model type
    using FactorType = ModelType::FactorType; // shortcut for model factor type
    using IndepFactorType = ModelType::IndependentFactorType; // shortcut for model indep factor type
    using FunctionID = ModelType::FunctionIdentifier; // shortcut for model function identifier type
    static constexpr LabelType s_nOccludedLabel = std::numeric_limits<LabelType>::max(); // last label value reserved for occluded pixels
    static constexpr size_t s_nMaxOrder = STEREOMATCH_MAX_FACT_ORDER; // used to limit internal static assignment array sizes
    static constexpr size_t s_nMaxCliqueAssign = 1<<s_nMaxOrder; // used to limit internal static assignment array sizes
    static_assert(std::is_integral<IndexType>::value,"Graph index type must be integral");
    static_assert(std::is_integral<LabelType>::value,"Graph label type must be integral");
    static_assert(std::numeric_limits<IndexType>::max()>=std::numeric_limits<LabelType>::max(),"Graph index type max value must be greater than label type max value");

    /// implements an inference strategy for a multi-label graphical model with non-submudular + higher-order terms
    struct StereoGraphInference : public opengm::Inference<ModelType,AccumulatorType> {

        using VerboseVisitorType = opengm::visitors::VerboseVisitor<StereoGraphInference>;
        using EmptyVisitorType = opengm::visitors::EmptyVisitor<StereoGraphInference>;
        using TimingVisitorType = opengm::visitors::TimingVisitor<StereoGraphInference>;

        /// stereo graph inference algo parameter holder
        struct Parameter {
            enum InitLabelingType {DEFAULT_LABEL,RANDOM_LABEL,LOCALOPT_LABEL,EXPLICIT_LABEL};
            enum IterLabelOrderType {DEFAULT_ORDER,RANDOM_ORDER,EXPLICIT_ORDER};
            Parameter(size_t nMaxIterCount = 1000) :
                    m_nMaxIterCount(nMaxIterCount),
                    m_eInitLabelType(DEFAULT_LABEL),
                    m_eIterLabelOrderType(DEFAULT_ORDER),
                    m_nIterLabelOrderRandomSeed(0),
                    m_nInitLabelRandomSeed(0),
                    m_vLabelOrdering(),
                    m_vInitLabeling() {}
            size_t m_nMaxIterCount;
            InitLabelingType m_eInitLabelType;
            IterLabelOrderType m_eIterLabelOrderType;
            size_t m_nIterLabelOrderRandomSeed,m_nInitLabelRandomSeed;
            std::vector<LabelType> m_vLabelOrdering,m_vInitLabeling;
        };

        /// full constructor of the inference algorithm structures; the graphical model must have already been constructed prior to this call
        StereoGraphInference(const ModelType&, const std::vector<LabelType>& vRealLabels, const cv::Size& oGridSize, Parameter=Parameter());
        /// returns the name of this inference method, for debugging/identification purposes
        virtual std::string name() const override {return std::string("litiv-stereo-matcher");}
        /// returns a copy of the internal const reference to the graphical model to solve
        virtual const ModelType& graphicalModel() const override {return m_oGM;}
        /// redirects inference to the infer(TVisitor&) implementation, passing along an empty visitor
        virtual opengm::InferenceTermination infer() override {EmptyVisitorType visitor; return infer(visitor);}
        /// sets a labeling starting point for the inference (the iterator must be valid over its 'nNodes' next values)
        virtual void setStartingPoint(typename std::vector<LabelType>::const_iterator begin) override {
            lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oGM.numberOfVariables()),m_vRealLabels).empty(),"provided labeling possesses invalid/out-of-range labels");
            std::transform(begin,begin+m_oGM.numberOfVariables(),m_oLabeling.begin(),[&](const LabelType& nRealLabel){return std::distance(m_vRealLabels.begin(),std::find(m_vRealLabels.begin(),m_vRealLabels.end(),nRealLabel));});
        }
        /// sets a labeling starting point for the inference (the iterator must be valid over its 'nNodes' next values); label values are expected to already be in the internal [0,nLabels] interval
        virtual void setStartingPoint_raw(typename std::vector<LabelType>::const_iterator begin) {
            lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oGM.numberOfVariables()),m_vLabels).empty(),"provided labeling possesses invalid/out-of-range labels");
            std::copy(begin,begin+m_oGM.numberOfVariables(),m_oLabeling.begin());
        }
        /// returns the *internal* labeling, as modified by solving the graphical model inference problem
        virtual opengm::InferenceTermination arg(std::vector<LabelType>& oLabeling, const size_t n=1) const override {
            if(n==1) {
                lvAssert_(m_oLabeling.total()==m_oGM.numberOfVariables(),"mismatch between internal graph label count and labeling mat size");
                oLabeling.resize(m_oLabeling.total());
                std::copy(m_oLabeling.begin(),m_oLabeling.end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones!
                return opengm::InferenceTermination::NORMAL;
            }
            return opengm::InferenceTermination::UNKNOWN;
        }
        /// resets internal parameters, assuming that the structure of the graphical model has not changed
        virtual void reset();
        /// performs the actual inference, taking a visitor as argument for debug purposes
        template<typename TVisitor>
        opengm::InferenceTermination infer(TVisitor&);

    protected:
        const ModelType& m_oGM;
        const Parameter m_oParams;
        const cv::Size m_oGridSize;
        const std::vector<LabelType> m_vRealLabels; ///< contains all usable labels, as provided through the constructor
        std::vector<LabelType> m_vLabels; ///< contains all 'real' labels, plus the 'occluded' label
        std::vector<LabelType> m_vLabelOrdering; ///< contains the label ordering to use for each iteration (with 'internal' values instead of 'real' ones)
        cv::Mat_<LabelType> m_oLabeling; ///< contains the 'internal' labeling instead of the 'real' one
        using AssocCountType = ushort;
        cv::Mat_<AssocCountType> m_oAssocCounts;
        cv::Mat_<bool> m_oAssocMap;
        size_t m_nLabels;
        size_t m_nAlphaLabel;
        size_t m_nOrderingIdx;
    };

    /// full stereo graph matcher constructor; relies on provided parameters to build the graphical model base without assigning full factor costs
    StereoMatcher(const cv::Size& oImagesSize, const std::vector<LabelType>& vAllowedDisparities) :
            m_vRealLabels(lv::unique(vAllowedDisparities.begin(),vAllowedDisparities.end())),
            m_vLabels(lv::concat<LabelType>(m_vRealLabels,std::vector<LabelType>{s_nOccludedLabel})),
            m_oGridSize(oImagesSize) {
        lvAssert_(m_oGridSize.area()>1,"graph grid must have at least two nodes");
        lvAssert_(m_vRealLabels.size()>1,"graph must have at least two possible output labels");
        const size_t nRealLabels = m_vRealLabels.size();
        const size_t nRows = (size_t)m_oGridSize.height;
        const size_t nCols = (size_t)m_oGridSize.width;
        const size_t nNodes = (size_t)m_oGridSize.area();
        std::cout << "Constructing graphical model for stereo matching..." << std::endl;
        lv::StopWatch oLocalTimer;
        m_pGM = std::make_unique<ModelType>(SpaceType(nNodes,m_vLabels.size()));
        {
            std::cout << "\tadding visual similarity factor for each grid pixel..." << std::endl;
            // (unary costs will depend on input data of both images, so each pixel function is likely unique)
            m_vVisSimUnaryFuncIDs.resize(nNodes);
            const std::array<LabelType,1> aUnaryFuncDims = {nRealLabels+1}; // +1 to account for occluded px label
            for(size_t nRowIdx = 0; nRowIdx<nRows; ++nRowIdx) {
                for(size_t nColIdx = 0; nColIdx<nCols; ++nColIdx) {
                    const size_t nNodeIdx = nRowIdx*nCols+nColIdx;
                    const FunctionID nFID = m_pGM->addFunction(ExplicitFunction());
                    ExplicitFunction& vUnaryFunc = m_pGM->getFunction<ExplicitFunction>(nFID);
                    vUnaryFunc.resize(aUnaryFuncDims.begin(),aUnaryFuncDims.end());
                    const std::array<size_t,1> aNodeIndices = {nNodeIdx};
                    m_pGM->addFactorNonFinalized(nFID,aNodeIndices.begin(),aNodeIndices.end());
                    m_vVisSimUnaryFuncIDs[nNodeIdx] = nFID;
                }
            }
        }
        {
            std::cout << "\tadding label similarity factor for each grid pixel pair..." << std::endl;
            // note: current def w/ explicit function will require too much memory if using >>50 labels
#if STEREOMATCH_USE_SMOOTH_POTTS
            const FunctionID nPottsFID = m_pGM->addFunction(PottsFunction(nRealLabels+1,nRealLabels+1,STEREOMATCH_LBLSIM_COST_EQUAL,STEREOMATCH_LBLSIM_COST_NEQUAL));
            m_vaSmoothPairwFuncIDs.resize(nNodes,std::array<FunctionID,2>{nPottsFID,nPottsFID});
#else //(!STEREOMATCH_USE_SMOOTH_POTTS)
            const std::array<LabelType,2> aPairwiseFuncDims = {nRealLabels+1,nRealLabels+1}; // +1 to account for occluded px label
            m_vaSmoothPairwFuncIDs.resize(nNodes);
#endif //(!STEREOMATCH_USE_SMOOTH_POTTS)
            std::array<size_t,2> aNodeIndices;
            for(size_t nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
                for(size_t nColIdx=0; nColIdx<nCols; ++nColIdx) {
                    aNodeIndices[0] = nRowIdx*nCols+nColIdx;
                    if(nRowIdx+1<nRows) {
                        aNodeIndices[1] = (nRowIdx+1)*nCols+nColIdx;
#if !STEREOMATCH_USE_SMOOTH_POTTS
                        m_vaSmoothPairwFuncIDs[aNodeIndices[0]][0] = m_pGM->addFunction(ExplicitFunction());
                        ExplicitFunction& vVerticalPairwiseFunc = m_pGM->getFunction<ExplicitFunction>(m_vaSmoothPairwFuncIDs[aNodeIndices[0]][0]);
                        vVerticalPairwiseFunc.resize(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end());
#endif //(!STEREOMATCH_USE_SMOOTH_POTTS)
                        m_pGM->addFactorNonFinalized(m_vaSmoothPairwFuncIDs[aNodeIndices[0]][0],aNodeIndices.begin(),aNodeIndices.end());
                    }
                    if(nColIdx+1<nCols) {
                        aNodeIndices[1] = nRowIdx*nCols+nColIdx+1;
#if !STEREOMATCH_USE_SMOOTH_POTTS
                        m_vaSmoothPairwFuncIDs[aNodeIndices[0]][1] = m_pGM->addFunction(ExplicitFunction());
                        ExplicitFunction& vHorizontalPairwiseFunc = m_pGM->getFunction<ExplicitFunction>(m_vaSmoothPairwFuncIDs[aNodeIndices[0]][1]);
                        vHorizontalPairwiseFunc.resize(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end());
#endif //(!STEREOMATCH_USE_SMOOTH_POTTS)
                        m_pGM->addFactorNonFinalized(m_vaSmoothPairwFuncIDs[aNodeIndices[0]][1],aNodeIndices.begin(),aNodeIndices.end());
                    }
                }
            }
        }/*{
         // add 3rd order function and factors to the model (test)
            const std::array<LabelType,3> aHOEFuncDims = {nLabels,nLabels,nLabels};
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

        m_pGM->finalize();
        std::cout << "Model constructed in " << oLocalTimer.tock() << " second(s)." << std::endl;
        lv::gm::printModelInfo(*m_pGM);
    }

    /// stereo matcher function; solves the graph model to find pixel-level matches on epipolar lines, and returns disparity masks
    virtual void apply(const MatArrayIn& aImages, MatArrayOut& oMasks) override {
        for(size_t nImgIdx=0; nImgIdx<aImages.size(); ++nImgIdx)
            lvAssert__(aImages[nImgIdx].size()==m_oGridSize && aImages[nImgIdx].type()==CV_8UC1,"input image in array at index=%d had the wrong size/type",nImgIdx);
        const size_t nRealLabels = m_vRealLabels.size();
        const size_t nOccludedLabelIdx = nRealLabels;
        const size_t nRows = (size_t)m_oGridSize.height;
        const size_t nCols = (size_t)m_oGridSize.width;
        std::cout << "Updating graphical model energy terms based on input data..." << std::endl;
        lv::StopWatch oLocalTimer;
        //const std::array<LabelType,2> aPairwiseFuncDims = {nRealLabels+1,nRealLabels+1}; // +1 to account for occluded px label
        for(size_t nRowIdx = 0; nRowIdx<nRows; ++nRowIdx) {
            for(size_t nColIdx = 0; nColIdx<nCols; ++nColIdx) {
                const size_t nNodeIdx = nRowIdx*nCols+nColIdx;
                {
                    // update visual similarity unary term for each grid pixel
                    const FunctionID& nFID = m_vVisSimUnaryFuncIDs[nNodeIdx];
                    ExplicitFunction& vUnaryFunc = m_pGM->getFunction<ExplicitFunction>(nFID);
                    lvDbgAssert(vUnaryFunc.dimension()==1 && vUnaryFunc.size()==nRealLabels+1);
                    for(size_t nLabelIdx = 0; nLabelIdx<nRealLabels; ++nLabelIdx) {
                        const size_t nColOffsetIdx = (nColIdx<size_t(m_vLabels[nLabelIdx]))?0:nColIdx-size_t(m_vLabels[nLabelIdx]);
                        vUnaryFunc(nLabelIdx) = std::min(STEREOMATCH_VISSIM_COST_MAXTRUNC,std::abs((float)aImages[0].at<uchar>(nRowIdx,nColIdx)-(float)aImages[1].at<uchar>(nRowIdx,nColOffsetIdx)));
                    }
                    vUnaryFunc(nOccludedLabelIdx) = STEREOMATCH_VISSIM_COST_OCCLUDED;
                }
#if !STEREOMATCH_USE_SMOOTH_POTTS
                {
                    // update smoothness pairwise terms for each grid pixel
                    const std::array<FunctionID,2>& anFIDs = m_vaSmoothPairwFuncIDs[nNodeIdx];
                    const auto lPairwiseUpdater = [&](size_t nOrientIdx){
                        ExplicitFunction& vPairwiseFunc = m_pGM->getFunction<ExplicitFunction>(anFIDs[nOrientIdx]);
                        lvDbgAssert(vPairwiseFunc.dimension()==2 && vPairwiseFunc.size()==(nRealLabels+1)*(nRealLabels+1));
                        for(size_t nLabelIdx1 = 0; nLabelIdx1<nRealLabels; ++nLabelIdx1)
                            for(size_t nLabelIdx2 = 0; nLabelIdx2<nRealLabels; ++nLabelIdx2)
                                vPairwiseFunc(nLabelIdx1,nLabelIdx2) = (nLabelIdx1==nLabelIdx2)?STEREOMATCH_LBLSIM_COST_EQUAL:STEREOMATCH_LBLSIM_COST_NEQUAL;
                        for(size_t nLabelIdx = 0; nLabelIdx<nRealLabels; ++nLabelIdx) {
                            vPairwiseFunc(nOccludedLabelIdx,nLabelIdx) = STEREOMATCH_LBLSIM_COST_NEQUAL; // @@@ change later for vis-data-dependent energies?
                            vPairwiseFunc(nLabelIdx,nOccludedLabelIdx) = STEREOMATCH_LBLSIM_COST_NEQUAL;
                        }
                        vPairwiseFunc(nOccludedLabelIdx,nOccludedLabelIdx) = STEREOMATCH_LBLSIM_COST_EQUAL;
                    };
                    if(nRowIdx+1<nRows)
                        lPairwiseUpdater(0);
                    if(nColIdx+1<nCols)
                        lPairwiseUpdater(1);
                }
#endif //(!STEREOMATCH_USE_SMOOTH_POTTS)
            }
        }
        std::cout << "Energy terms update completed in " << oLocalTimer.tock() << " second(s)." << std::endl;
        std::vector<LabelType> vOutputLabels(m_pGM->numberOfVariables());
        StereoGraphInference::Parameter p;
        StereoGraphInference oSolver(*m_pGM,m_vRealLabels,m_oGridSize,p);
        StereoGraphInference::VerboseVisitorType visitor;
        oSolver.infer(visitor);
        oSolver.arg(vOutputLabels);
        std::cout << "Inference completed in " << oLocalTimer.tock() << " second(s)." << std::endl;
        oMasks[0].create(m_oGridSize);
        lvAssert_((size_t)m_vRealLabels.back()<(size_t)s_nOccludedLabel,"label values will not fit in output mat depth");
        for(size_t nRowIdx=0; nRowIdx<(size_t)m_oGridSize.height; ++nRowIdx) {
            for(size_t nColIdx=0; nColIdx<(size_t)m_oGridSize.width; ++nColIdx) {
                const size_t nNodeIdx = nRowIdx*((size_t)m_oGridSize.width)+nColIdx;
                const LabelType nNodeLabel = vOutputLabels[nNodeIdx];
                oMasks[0](nRowIdx,nColIdx) = ((size_t)nNodeLabel>=m_vRealLabels.size())?s_nOccludedLabel:m_vRealLabels[nNodeLabel];
            }
        }
    }
    /// returns the (maximum) number of labels used in the output masks, or 0 if it cannot be predetermined
    virtual size_t getMaxLabelCount() const override {return m_vLabels.size();}
    /// returns the list of labels used in the output masks, or an empty array if it cannot be predetermined
    virtual const std::vector<LabelType>& getLabels() const override {return m_vLabels;}

protected:
    //std::unique_ptr<StereoGraphInference> m_pInf;
    std::unique_ptr<ModelType> m_pGM; ///< should exist as long as inf algo exists; holds the graphical model
    const std::vector<LabelType> m_vRealLabels; ///< contains all usable labels, as provided through the constructor
    const std::vector<LabelType> m_vLabels;  ///< contains all 'real' labels, plus the 'occluded' label
    const cv::Size m_oGridSize;
    std::vector<FunctionID> m_vVisSimUnaryFuncIDs;
    std::vector<std::array<FunctionID,2>> m_vaSmoothPairwFuncIDs;

};

inline StereoMatcher::StereoGraphInference::StereoGraphInference(const ModelType& oGM, const std::vector<LabelType>& vRealLabels, const cv::Size& oGridSize, Parameter p) :
        m_oGM(oGM),m_oParams(p),m_oGridSize(oGridSize),m_vRealLabels(lv::unique(vRealLabels.begin(),vRealLabels.end())),m_vLabels(lv::concat<LabelType>(m_vRealLabels,std::vector<LabelType>{s_nOccludedLabel})),m_nLabels(m_vLabels.size()) {
    lvAssert_(m_vRealLabels.size()>1,"graph must have at least two output labels");
    lvAssert_(m_oGridSize.area()>1,"graph must have at least two nodes");
    lvAssert_(m_oGM.numberOfFactors()>0,"graph had no valid factors");
    for(size_t nFactIdx=0; nFactIdx<m_oGM.numberOfFactors(); ++nFactIdx)
        lvAssert__(m_oGM[nFactIdx].numberOfVariables()<=s_nMaxOrder,"graph had some factors of order %d (max allowed is %d)",(int)m_oGM[nFactIdx].numberOfVariables(),(int)s_nMaxOrder);
    lvAssert_(m_oGM.numberOfVariables()>0 && m_oGM.numberOfVariables()==(IndexType)m_oGridSize.area(),"graph node count must match grid size");
    for(size_t nNodeIdx=1; nNodeIdx<m_oGM.numberOfVariables(); ++nNodeIdx)
        lvAssert_(m_oGM.numberOfLabels(nNodeIdx)==m_nLabels,"graph nodes must all have the same number of labels");
    m_oLabeling.create(m_oGridSize);
    lvAssert_(std::numeric_limits<AssocCountType>::max()>m_oGridSize.width,"grid width is too large for association counter type");
    m_oAssocCounts.create(m_oGridSize);
    const int anAssocMapDims[] = {m_oGridSize.height,m_oGridSize.width,m_oGridSize.width};
    m_oAssocMap.create(3,anAssocMapDims);
    reset();
}

inline void StereoMatcher::StereoGraphInference::reset() {
    if(m_oParams.m_eInitLabelType == Parameter::RANDOM_LABEL) {
        srand(m_oParams.m_nInitLabelRandomSeed);
        for(size_t nNodeIdx=0; nNodeIdx<m_oGM.numberOfVariables(); ++nNodeIdx) {
            lvAssert_(m_oGM.numberOfLabels(nNodeIdx)==m_nLabels,"graph nodes must all have the same number of labels");
            m_oLabeling(nNodeIdx) = rand()%m_nLabels;
        }
    }
    else if(m_oParams.m_eInitLabelType == Parameter::LOCALOPT_LABEL) {
        std::fill(m_oLabeling.begin(),m_oLabeling.end(),(LabelType)0);
        for(size_t nFactIdx=0; nFactIdx<m_oGM.numberOfFactors(); ++nFactIdx) {
            if(m_oGM[nFactIdx].numberOfVariables()==1) {
                LabelType nEvalLabel = 0;
                ValueType fOptimalEnergy = m_oGM[nFactIdx](&nEvalLabel);
                for(nEvalLabel=1; nEvalLabel<m_oGM.numberOfLabels(nFactIdx); ++nEvalLabel) {
                    if(AccumulatorType::bop(m_oGM[nFactIdx](&nEvalLabel),fOptimalEnergy)) {
                        fOptimalEnergy = m_oGM[nFactIdx](&nEvalLabel);
                        m_oLabeling(m_oGM.variableOfFactor(nFactIdx,0)) = nEvalLabel;
                    }
                }
            }
        }
    }
    else if(m_oParams.m_eInitLabelType == Parameter::EXPLICIT_LABEL) {
        lvAssert_(m_oParams.m_vInitLabeling.size()==m_oGM.numberOfVariables(),"graph node count and initialization labeling size mismatch");
        lvAssert_(lv::filter_out(m_oParams.m_vInitLabeling,m_vLabels).empty(),"some labels in the initialization labeling were invalid/out-of-range");
        std::transform(m_oParams.m_vInitLabeling.begin(),m_oParams.m_vInitLabeling.end(),m_oLabeling.begin(),[&](const LabelType& nLabel){return std::distance(m_vLabels.begin(),std::find(m_vLabels.begin(),m_vLabels.end(),nLabel));});
    }
    else
        std::fill(m_oLabeling.begin(),m_oLabeling.end(),0);
    m_oAssocCounts = (AssocCountType)0;
    m_oAssocMap = false;
    for(size_t nRowIdx=0; nRowIdx<(size_t)m_oGridSize.height; ++nRowIdx) {
        for(size_t nColIdx=0; nColIdx<(size_t)m_oGridSize.width; ++nColIdx) {
            const LabelType& nCurrNodeLabel = m_oLabeling(nRowIdx,nColIdx);
            if(nColIdx>=(size_t)nCurrNodeLabel) {
                const size_t nAssocNodeColIdx = nColIdx-nCurrNodeLabel;
                ++m_oAssocCounts(nRowIdx,nAssocNodeColIdx);
                m_oAssocMap(nRowIdx,nColIdx,nAssocNodeColIdx) = true;
            }
        }
    }
    m_vLabelOrdering.resize(m_nLabels);
    if(m_oParams.m_eIterLabelOrderType == Parameter::RANDOM_ORDER) {
        std::iota(m_vLabelOrdering.begin(),m_vLabelOrdering.end(),0);
        std::mt19937 oGen(m_oParams.m_nIterLabelOrderRandomSeed);
        std::shuffle(m_vLabelOrdering.begin(),m_vLabelOrdering.end(),oGen);
    }
    else if(m_oParams.m_eIterLabelOrderType == Parameter::EXPLICIT_ORDER) {
        lvAssert_(m_oParams.m_vLabelOrdering.size()==m_nLabels,"label order array did not contain all labels");
        lvAssert_(lv::unique(m_oParams.m_vLabelOrdering.begin(),m_oParams.m_vLabelOrdering.end())==m_vLabels,"label order array did not contain all labels");
        m_vLabelOrdering = lv::indices_of(m_oParams.m_vLabelOrdering,m_vLabels);
    }
    else
        std::iota(m_vLabelOrdering.begin(),m_vLabelOrdering.end(),0);
    m_nOrderingIdx = 0;
    m_nAlphaLabel = m_vLabelOrdering[m_nOrderingIdx];
}

template<typename TVisitor>
inline opengm::InferenceTermination StereoMatcher::StereoGraphInference::infer(TVisitor& oVisitor) {
    const size_t nTotNodeCount = m_oGM.numberOfVariables();
    lvAssert_(nTotNodeCount==m_oLabeling.total(),"graph node count and labeling mat size mismatch");
    kolmogorov::qpbo::QPBO<ValueType> oBinaryEnergyMinimizer(nTotNodeCount,0);
    std::array<ValueType,s_nMaxCliqueAssign> vCliqueCoeffs;
    std::array<LabelType,s_nMaxOrder> vCliqueLabels;
    std::array<typename HigherOrderEnergy<ValueType,s_nMaxOrder>::VarId,s_nMaxOrder> aTermEnergyLUT;
    size_t nIter = 0, nConsecUnchangedLabels = 0;
    oVisitor.begin(*this);
    // each iter below is an alpha-exp move based on A. Fix's primal-dual energy minimization method for higher-order MRFs
    while(++nIter<=m_oParams.m_nMaxIterCount && nConsecUnchangedLabels<m_nLabels) {
        HigherOrderEnergy<ValueType,s_nMaxOrder> oHigherOrderEnergyReducer;
        oHigherOrderEnergyReducer.AddVars(nTotNodeCount);
        for(size_t nFactIdx=0; nFactIdx<m_oGM.numberOfFactors(); ++nFactIdx) {
            const size_t nCurrFactOrder = m_oGM[nFactIdx].numberOfVariables();
            if(nCurrFactOrder==1) {
                const size_t nNodeIdx = m_oGM[nFactIdx].variableIndex(0);
                const LabelType nNodeInitLabel = m_oLabeling(nNodeIdx);
                const ValueType fNodeInitVisSimEnergy = m_oGM[nFactIdx](&nNodeInitLabel);
                const ValueType fNodeModifVisSimEnergy = m_oGM[nFactIdx](&m_nAlphaLabel);
                lvDbgAssert(m_oAssocMap.step.p[0]==(size_t)m_oGridSize.width*m_oGridSize.width && m_oAssocMap.step.p[1]==(size_t)m_oGridSize.width && m_oAssocMap.step.p[2]==1);
                lvDbgAssert(m_oAssocCounts.step.p[0]==(size_t)m_oGridSize.width*sizeof(AssocCountType) && m_oAssocCounts.step.p[1]==sizeof(AssocCountType));
                const size_t nColIdx = nNodeIdx%m_oGridSize.width;
                const AssocCountType nInitAssocCount = ((AssocCountType*)m_oAssocCounts.data)[nNodeIdx];
                const AssocCountType nModifAssocCount = nColIdx>=(size_t)nNodeInitLabel?((AssocCountType*)m_oAssocCounts.data)[nNodeIdx-m_nAlphaLabel]:(STEREOMATCH_UNIQUE_COST_OOBASSOC); // @@@@ replace const by something more... wise
                const ValueType fNodeInitOverAssocEnergy = (ValueType)(nInitAssocCount*(ValueType)STEREOMATCH_UNIQUE_COST_OVERASSOC);
                const ValueType fNodeModifOverAssocEnergy = (ValueType)(nModifAssocCount*(ValueType)STEREOMATCH_UNIQUE_COST_OVERASSOC);
                oHigherOrderEnergyReducer.AddUnaryTerm(nNodeIdx,(fNodeModifVisSimEnergy+fNodeModifOverAssocEnergy)-(fNodeInitVisSimEnergy+fNodeInitOverAssocEnergy)); // @@@@ check sign of term?
            }
            else if(nCurrFactOrder>1) {
                // rewritten from Fix's HigherOrderEnergy<R,D>::AddClique(vars,etable)
                const size_t nAssignCount = 1<<nCurrFactOrder;
                std::fill_n(vCliqueCoeffs.begin(),nAssignCount,(ValueType)0);
                for(size_t nAssignIdx=0; nAssignIdx<nAssignCount; ++nAssignIdx) {
                    for(size_t nNodeIdx=0; nNodeIdx<nCurrFactOrder; ++nNodeIdx)
                        vCliqueLabels[nNodeIdx] = (nAssignIdx&(1<<nNodeIdx))?m_nAlphaLabel:m_oLabeling(m_oGM[nFactIdx].variableIndex(nNodeIdx));
                    const ValueType fCurrAssignEnergy = m_oGM[nFactIdx](vCliqueLabels.begin());
                    for(size_t nAssignSubsetIdx=1; nAssignSubsetIdx<nAssignCount; ++nAssignSubsetIdx) {
                        if(!(nAssignIdx&~nAssignSubsetIdx)) {
                            int nParityBit = 0;
                            for(size_t nNodeIdx=0; nNodeIdx<nCurrFactOrder; ++nNodeIdx)
                                nParityBit ^= (((nAssignIdx^nAssignSubsetIdx)&(1<<nNodeIdx))!=0);
                            vCliqueCoeffs[nAssignSubsetIdx] += nParityBit?-fCurrAssignEnergy:fCurrAssignEnergy;
                        }
                    }
                }
                for(size_t nAssignSubsetIdx=1; nAssignSubsetIdx<nAssignCount; ++nAssignSubsetIdx) {
                    int nCurrTermDegree = 0;
                    for(size_t nNodeIdx=0; nNodeIdx<nCurrFactOrder; ++nNodeIdx)
                        if(nAssignSubsetIdx & (1<<nNodeIdx))
                            aTermEnergyLUT[nCurrTermDegree++] = m_oGM[nFactIdx].variableIndex(nNodeIdx);
                    std::sort(aTermEnergyLUT.begin(),aTermEnergyLUT.begin()+nCurrTermDegree);
                    oHigherOrderEnergyReducer.AddTerm(vCliqueCoeffs[nAssignSubsetIdx],nCurrTermDegree,aTermEnergyLUT.data());
                }
            }
        }
        oBinaryEnergyMinimizer.Reset();
        oHigherOrderEnergyReducer.ToQuadratic(oBinaryEnergyMinimizer);
        //@@@@@@ oBinaryEnergyMinimizer.AddPairwiseTerm(nodei,nodej,val00,val01,val10,val11);
        oBinaryEnergyMinimizer.Solve();
        size_t nChangedLabelings = 0;
        for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
            if(oBinaryEnergyMinimizer.GetLabel(nNodeIdx)!=0) { // label changed
                const LabelType nOldLabel = m_oLabeling(nNodeIdx);
                m_oLabeling(nNodeIdx) = m_nAlphaLabel;
                //if(nOldLabel!=@@@occluded)
                // @@@ adjust assoc map here
                ++nChangedLabelings;
            }
        }
        //ValueType energy2 = m_oGM.evaluate(m_vNodeLabels);
        nConsecUnchangedLabels = (nChangedLabelings>0)?0:nConsecUnchangedLabels+1;
        // @@@@ order of future moves can be influenced by labels that cause the most changes? (but only late, to avoid bad local minima?)
        //oVisitor(*this,energy2,this->bound(),"alpha",m_nAlphaLabel);
        m_nAlphaLabel = m_vLabelOrdering[(++m_nOrderingIdx%=m_nLabels)];
        if(oVisitor(*this)!=opengm::visitors::VisitorReturnFlag::ContinueInf)
            break;
    }
    oVisitor.end(*this);
    return opengm::InferenceTermination::NORMAL;
}
